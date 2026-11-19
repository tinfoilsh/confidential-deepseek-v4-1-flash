#!/usr/bin/env python3
"""CPU-only patch check against a pristine checkout of the pinned vLLM source.

Usage: python scripts/test_bounded_chat_input.py /path/to/vllm
Requires Git, GNU patch and pydantic 2. Does not import vLLM or exercise GPUs.
"""

from __future__ import annotations

import __future__
import ast
import asyncio
import hashlib
import runpy
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Annotated, Any
from unittest.mock import Mock

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator


PIN = "db9527a46873454610df6dbedf79a36d6bf1a7f6"
SOURCE_HASHES = {
    "protocol.py": "4bdc71f668e4820d8ae5bae2a153fed0296b430cd6353ace8c0c27cfb982ebc1",
    "serving.py": "532dadea845c77a5a79011027b9989987c3e32467614e993f439f29c924de002",
}
CHAT = Path("vllm/entrypoints/openai/chat_completion")
IMAGE_SITE = Path("usr/local/lib/python3.12/dist-packages")
PATCH = (
    Path(__file__).resolve().parents[1] / "b300/patches/0013-bounded-chat-input.patch"
)


EXCEPTIONS_HASH = "22fe819c7ef10438f740c251d81af204871a04c3aa3eb1fd0e68604dc4c2146f"


@dataclass
class ErrorResponse:
    message: str
    param: str
    status: int = 400


def load_guard(source, validation_error):
    """Execute the real field/validator and serving method with minimal doubles."""
    protocol = ast.parse((source / CHAT / "protocol.py").read_text())
    real_request = next(
        node
        for node in protocol.body
        if isinstance(node, ast.ClassDef) and node.name == "ChatCompletionRequest"
    )
    names = {
        "max_input_tokens",
        "n",
        "use_beam_search",
        "truncate_prompt_tokens",
        "kv_transfer_params",
        "ec_transfer_params",
        "vllm_xargs",
    }
    fields_and_validator = [
        node
        for node in real_request.body
        if (isinstance(node, ast.AnnAssign) and node.target.id in names)
        or (
            isinstance(node, ast.FunctionDef) and node.name == "_validate_bounded_input"
        )
    ]
    test_request_source = ast.parse(
        "class ChatCompletionRequest(BaseModel):\n"
        "    model_config = ConfigDict(extra='allow')\n"
        "    stream: bool = False\n"
        "    messages: list[dict] = []\n"
    )
    test_request_source.body[0].body.extend(fields_and_validator)
    serving = ast.parse((source / CHAT / "serving.py").read_text())
    create = next(
        node
        for node in ast.walk(serving)
        if isinstance(node, ast.AsyncFunctionDef)
        and node.name == "_create_chat_completion"
    )
    bindings = {
        "Any": Any,
        "Annotated": Annotated,
        "BaseModel": BaseModel,
        "ConfigDict": ConfigDict,
        "Field": Field,
        "model_validator": model_validator,
        "VLLMValidationError": validation_error,
        "ErrorResponse": ErrorResponse,
        "_INT64_MAX": 2**63 - 1,
        "BeamSearchParams": type("BeamSearchParams", (), {}),
        "RequestResponseMetadata": SimpleNamespace,
        "get_max_tokens": lambda *args, **kwargs: 12,
        "_get_mm_token_counts": lambda _: {},
    }
    exec(
        compile(test_request_source, "protocol.py", "exec", dont_inherit=True), bindings
    )
    module = ast.Module(body=[create], type_ignores=[])
    exec(
        compile(
            module, "serving.py", "exec", flags=__future__.annotations.compiler_flag
        ),
        bindings,
    )
    return bindings["ChatCompletionRequest"], bindings[create.name]


def check_protocol(request, validation_error):
    assert request().max_input_tokens is None
    try:
        request(max_input_tokens=None)
    except validation_error as err:
        assert err.parameter == "max_input_tokens", err
    else:
        raise AssertionError("accepted explicit null max_input_tokens")
    for value in (True, 0, -1, 1.0, "1", 2**63):
        try:
            request(max_input_tokens=value)
        except ValidationError as err:
            assert err.errors()[0]["loc"] == ("max_input_tokens",), err
            continue
        raise AssertionError(f"accepted invalid max_input_tokens={value!r}")
    for field, value in (
        ("n", 2),
        ("n", None),
        ("use_beam_search", True),
        ("truncate_prompt_tokens", 1),
        ("truncate_prompt_tokens", -1),
        ("kv_transfer_params", {"prompt_token_ids": [1]}),
        ("kv_transfer_params", {}),
        ("ec_transfer_params", {}),
        ("vllm_xargs", {}),
        ("best_of", 1),
        ("prompt_embeds", "not used"),
    ):
        request(**{field: value})  # Do not add constraints to uncapped requests.
        try:
            request(max_input_tokens=3, **{field: value})
        except validation_error as err:
            assert err.parameter == field, err
        else:
            raise AssertionError(f"accepted {field} with a cap")


async def check_serving(request, create):
    messages = [{"role": "user", "content": "one raw word"}]
    rendered = {"type": "token", "prompt_token_ids": [10, 11, 12, 13]}
    # The renderer stand-in provides IDs already expanded around a four-token
    # image placeholder span. The cap is on all six IDs, not just the two
    # text IDs or a single placeholder for the image.
    image = {
        "type": "multimodal",
        "prompt_token_ids": [10, 90, 90, 90, 90, 13],
        "mm_hashes": {"image": ["image-hash"]},
        "mm_placeholders": {"image": [SimpleNamespace(offset=1, length=4)]},
    }
    for stream in (False, True):
        for limit, inputs, should_reject in (
            (4, [rendered], False),  # Exact-bound acceptance, no truncation.
            (3, [rendered], True),  # Including hypothetical template/schema overhead.
            (None, [rendered], False),  # Uncapped behavior unchanged.
            (6, [image], False),
            (5, [image], True),
            (None, [image], False),
            (6, [{**image, "encoder_prompt_token_ids": [1]}], True),
            (6, [{**image, "encoder_prompt": "other prompt"}], True),
            (6, [{**image, "mm_placeholders": None}], True),
            (6, [{**image, "mm_placeholders": {"image": []}}], True),
            (6, [{**image, "mm_hashes": {"image": ["a", "b"]}}], True),
            (6, [{**image, "mm_hashes": {"image": []}}], True),
            (6, [{**image, "mm_hashes": {"image": ["img"], "video": ["v"]}}], True),
            (6, [{**image, "mm_placeholders": {"audio": [object()]}}], True),
            (
                6,
                [
                    {
                        **image,
                        "mm_hashes": {"audio": ["audio-hash"]},
                        "mm_placeholders": {"audio": [object()]},
                    }
                ],
                True,
            ),
            (4, [{"type": "embeds", "prompt_token_ids": [1]}], True),
            (4, [{"type": "multimodal", "prompt_token_ids": [1]}], True),
            (4, [{"type": "enc_dec", "prompt_token_ids": [1]}], True),
            (4, [{"prompt_token_ids": [1]}], True),
            (4, [{"type": "token"}], True),
            (10, [rendered, image], True),
            (4, [], True),
        ):
            original_ids = [item.get("prompt_token_ids", []).copy() for item in inputs]
            kwargs = {} if limit is None else {"max_input_tokens": limit}
            cap = request(stream=stream, messages=messages, **kwargs)
            req = SimpleNamespace(
                **cap.model_dump(),
                request_id="id",
                max_completion_tokens=12,
                include_reasoning=False,
                _grammar_from_parser=False,
                to_sampling_params=Mock(return_value=object()),
            )
            generate = Mock(return_value=object())
            full = Mock()

            async def full_response(*args, **kwargs):
                full(*args, **kwargs)
                return "full"

            async def render(received):
                assert received is req
                return messages, inputs

            serve = SimpleNamespace(
                renderer=SimpleNamespace(tokenizer=object()),
                parser_cls=None,
                _effective_chat_template_kwargs=Mock(return_value={}),
                render_chat_request=render,
                create_error_response=ErrorResponse,
                _base_request_id=Mock(return_value="id"),
                _maybe_get_adapters=Mock(),
                models=SimpleNamespace(model_name=Mock(return_value="model")),
                model_config=SimpleNamespace(max_model_len=100),
                engine_client=SimpleNamespace(generate=generate),
                default_sampling_params={},
                override_max_tokens=None,
                _get_data_parallel_rank=Mock(),
                _extract_prompt_len=lambda item: len(item["prompt_token_ids"]),
                _extract_prompt_components=lambda item: SimpleNamespace(
                    token_ids=item["prompt_token_ids"]
                ),
                _log_inputs=Mock(),
                _get_session_id=Mock(),
                _get_priority=Mock(),
                chat_completion_stream_generator=Mock(return_value="stream"),
                chat_completion_full_generator=full_response,
            )
            result = await create(serve, req)
            if should_reject:
                assert isinstance(result, ErrorResponse), result
                assert result.status == 400 and result.param == "max_input_tokens"
                generate.assert_not_called()
                serve.chat_completion_stream_generator.assert_not_called()
                full.assert_not_called()
                serve._base_request_id.assert_not_called()
            else:
                assert result == ("stream" if stream else "full"), result
                generate.assert_called_once()
                assert generate.call_args.args[0] is inputs[0]
            assert [item.get("prompt_token_ids", []) for item in inputs] == original_ids


def main(checkout):
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)

        def stage(path, expected_hash):
            target = root / IMAGE_SITE / path
            target.parent.mkdir(parents=True, exist_ok=True)
            if (checkout / ".git").exists():
                with target.open("wb") as out:
                    subprocess.run(
                        ["git", "show", f"{PIN}:{path.as_posix()}"],
                        cwd=checkout,
                        stdout=out,
                        check=True,
                    )
            else:
                target.write_bytes((checkout / path).read_bytes())
            if hashlib.sha256(target.read_bytes()).hexdigest() != expected_hash:
                raise ValueError(f"{path} does not match pinned source {PIN}")
            return target

        for file, expected_hash in SOURCE_HASHES.items():
            stage(CHAT / file, expected_hash)
        exceptions = stage(Path("vllm/exceptions.py"), EXCEPTIONS_HASH)
        validation_error = runpy.run_path(str(exceptions))["VLLMValidationError"]
        subprocess.run(
            [
                "patch",
                "-p1",
                "--batch",
                "--forward",
                "--no-backup-if-mismatch",
                "--fuzz=0",
            ],
            cwd=root,
            input=PATCH.read_bytes(),
            check=True,
        )
        request, create = load_guard(root / IMAGE_SITE, validation_error)
        check_protocol(request, validation_error)
        asyncio.run(check_serving(request, create))
    print("PASS: patch application, Pydantic validation and mocked generation boundary")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(f"Usage: {sys.argv[0]} /path/to/pinned/vllm/source")
    main(Path(sys.argv[1]).resolve())
