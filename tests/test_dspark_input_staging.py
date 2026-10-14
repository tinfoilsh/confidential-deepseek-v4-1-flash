"""Check private input copies and DeepSeek text-path selection on CUDA."""

import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.v1.worker.gpu.cc_input_staging import private_tensor_h2d, supports_text_staging
from vllm.v1.worker.gpu.ec_connector import ECConnector
from vllm.v1.worker.gpu.kv_connector import NO_OP_KV_CONNECTOR
from vllm.v1.worker.gpu.sample.sampler import BadWordsState, LogitBiasState, PenaltiesState

from test_dspark_readback import supported_worker

LENGTHS = (1, 8, 64)
MAX_REQUESTS = 64
OFFSET_SENTINEL = -7


@pytest.fixture
def device():
    assert os.environ.get("VLLM_DISABLE_UVA") == "1"
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    with torch.cuda.device(0):
        yield torch.device("cuda:0")
        torch.cuda.synchronize()


@pytest.mark.parametrize("length", LENGTHS)
def test_index_copy_owns_values_after_scheduler_mutation(device, length):
    source = np.arange(length, dtype=np.intp)[::-1].copy()
    expected = torch.tensor(source, dtype=torch.int32)
    result = private_tensor_h2d(source, device=device, dtype=torch.int32)
    source[:] = OFFSET_SENTINEL
    torch.testing.assert_close(result.cpu(), expected, rtol=0, atol=0)


def test_full_cumulative_output_preserves_pointer_and_caller_values(device):
    source = np.arange(MAX_REQUESTS + 1, dtype=np.int32)
    output = torch.empty(source.shape, device=device, dtype=torch.int32)
    pointer = output.data_ptr()
    for length in LENGTHS:
        source[length + 1:] = source[length]
        expected = torch.tensor(source.copy())
        assert private_tensor_h2d(source, out=output) is output
        source[:] = OFFSET_SENTINEL
        assert output.data_ptr() == pointer
        torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
        source[:] = np.arange(source.size)


def test_offset_output_fallback_preserves_surrounding_storage(device):
    source = np.arange(len(LENGTHS), dtype=np.int32)
    backing = torch.full((len(source) + 2,), OFFSET_SENTINEL, device=device,
                         dtype=torch.int32)
    result = private_tensor_h2d(source, out=backing[1:-1])
    torch.testing.assert_close(result.cpu(), torch.tensor(source), rtol=0, atol=0)
    assert backing[0].item() == backing[-1].item() == OFFSET_SENTINEL


def text_runtime(device):
    worker = supported_worker(device)
    runner = worker.model_runner
    runner.parallel_config = worker.parallel_config
    for name in ("enable_dbo", "use_ubatching", "enable_batch_sharded_sampling"):
        setattr(runner.parallel_config, name, False)
    runner.lora_config = None
    for name in ("is_pooling_model", "is_encoder_decoder", "use_pp", "use_dcp"):
        setattr(runner, name, False)
    runner.is_first_pp_rank = runner.is_last_pp_rank = True
    runner.dp_size = runner.dcp_size = 1
    runner.model_config.logits_processors = None
    runner.model_config.enable_prompt_embeds = False
    runner.sampler.logits_processors = [object.__new__(kind) for kind in
                                       (LogitBiasState, PenaltiesState, BadWordsState)]
    runner.input_buffers = object()
    runner.speculator.target_input_buffers = runner.input_buffers
    runner.speculator.input_buffers = object()
    runner.kv_connector = NO_OP_KV_CONNECTOR
    runner.ec_connector = ECConnector()
    runner.model_state.prompt_embeds_state = None
    runner.model_state.supports_mm_inputs = True
    runner.supports_mm_inputs = runner.uses_inputs_embeds = True
    runner.encoder_cache = SimpleNamespace(mm_features={"request": []})
    runner.main_stream = torch.cuda.current_stream()
    for name in ("pcp_manager", "ubatch_runner", "batch_sharder", "fast_prefill",
                 "aux_output_connector", "pooling_runner", "pp_handler"):
        setattr(runner, name, None)
    batch = SimpleNamespace(num_tokens=1, req_ids=["request"])
    request = SimpleNamespace(mm_features=[], prompt_embeds=None, prompt_is_token_ids=None)
    scheduler = SimpleNamespace(num_scheduled_tokens={"request": 1},
                                scheduled_encoder_inputs={}, scheduled_new_reqs=[request])
    return runner, scheduler, batch


def test_text_guard_rejects_images_cpu_offload_and_nondefault_speculation(device):
    runner, scheduler, batch = text_runtime(device)
    assert supports_text_staging(runner, scheduler, batch, 0)
    runner.encoder_cache.mm_features["request"] = [object()]
    assert not supports_text_staging(runner, scheduler, batch, 0)
    runner.encoder_cache.mm_features["request"] = []
    runner.kv_connector = object()
    assert not supports_text_staging(runner, scheduler, batch, 0)
    runner.kv_connector = NO_OP_KV_CONNECTOR
    runner.speculative_config.enable_adaptive_verification = False
    assert not supports_text_staging(runner, scheduler, batch, 0)
