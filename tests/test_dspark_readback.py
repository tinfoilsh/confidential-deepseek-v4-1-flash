"""Exercise deferred output with real CUDA state updates and consumer threads."""

from concurrent.futures import ThreadPoolExecutor
import os
from threading import current_thread
from types import SimpleNamespace

import pytest
import torch

from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.worker.gpu.async_utils import AsyncOutput
from vllm.v1.worker.gpu.cc_readback import (
    CCReadbackPolicy, _DeferredOutput, _supported_config,
)
from vllm.v1.worker.gpu.input_batch import get_num_sampled_and_rejected, post_update
from vllm.v1.worker.gpu.sample.output import SamplerOutput

ROWS = 3
DEPTH = 5
WIDTH = DEPTH + 1
STEPS = 12
HISTORY = 256
TOKEN_STRIDE = 32
JOIN_TIMEOUT_SECONDS = 60
COUNTS = (0, 1, WIDTH)
MAX_REQUESTS = 64
MAX_CONTEXT = 1_048_576
TENSOR_PARALLEL_SIZE = 4


@pytest.fixture
def runtime():
    assert os.environ.get("VLLM_DISABLE_UVA") == "1"
    assert torch.cuda.is_available()
    assert torch.cuda.device_count() == 1
    device = torch.device("cuda:0")
    with torch.cuda.device(device):
        runner = SimpleNamespace(device=device, main_stream=torch.cuda.current_stream(),
                                 output_copy_stream=torch.cuda.Stream())
        yield runner, CCReadbackPolicy(runner)
        torch.cuda.synchronize()


def make_step(runner, policy, consumer, step, *, reply=True, optional=False):
    values = torch.arange(ROWS * WIDTH, device=runner.device, dtype=torch.int64)
    values = values.reshape(ROWS, WIDTH) + step * TOKEN_STRIDE
    counts = torch.full((ROWS,), WIDTH, device=runner.device, dtype=torch.int32)
    mapping = torch.arange(ROWS, device=runner.device, dtype=torch.int32)
    seq_lens = torch.tensor([1, HISTORY, HISTORY], device=runner.device,
                            dtype=torch.int32)
    prefill = torch.full((ROWS,), HISTORY, device=runner.device, dtype=torch.int32)
    logits = torch.arange(ROWS + 1, device=runner.device, dtype=torch.int32) * WIDTH
    counts[1] = COUNTS[1]
    counts, rejected = get_num_sampled_and_rejected(counts, seq_lens, logits,
                                                   mapping, prefill)
    sampler = SamplerOutput(values, None, torch.zeros_like(counts) if optional else None,
                            counts, rejected)
    output = ModelRunnerOutput([f"request-{step}-{row}" for row in range(ROWS)],
                               {f"request-{step}-{row}": row for row in range(ROWS)})
    token = policy.context.set((reply, consumer))
    try:
        result = policy.make_output(output, sampler, counts, runner.main_stream,
                                    runner.output_copy_stream, False, None)
    finally:
        policy.context.reset(token)
    history = torch.zeros((ROWS, HISTORY), device=runner.device, dtype=torch.int64)
    last = torch.zeros(ROWS, device=runner.device, dtype=torch.int64)
    lengths = torch.zeros(ROWS, device=runner.device, dtype=torch.int32)
    computed = lengths.clone()
    post_update(mapping, computed, last, None, values, counts, rejected, logits,
                history, lengths)
    expected = [[step * TOKEN_STRIDE + row * WIDTH + index
                 for index in range(count)] for row, count in enumerate(COUNTS)]
    return result, expected, history, lengths


def test_consumer_retains_multiple_outputs_across_request_state_updates(runtime):
    runner, policy = runtime
    with ThreadPoolExecutor(max_workers=1) as executor:
        consumer = executor.submit(current_thread).result(JOIN_TIMEOUT_SECONDS)
        pending = [make_step(runner, policy, consumer, step) for step in range(STEPS)]
        assert len(policy.pending) == STEPS
        for result, expected, history, lengths in pending:
            assert type(result) is _DeferredOutput
            output = executor.submit(result.get_output).result(JOIN_TIMEOUT_SECONDS)
            assert output.sampled_token_ids == expected
            assert lengths.cpu().tolist() == list(COUNTS)
            for row, tokens in enumerate(expected):
                assert history[row, :len(tokens)].cpu().tolist() == tokens
            assert executor.submit(result.get_output).result(JOIN_TIMEOUT_SECONDS) is output
    assert not policy.pending


def test_wrong_consumer_is_rejected_without_consuming_the_output(runtime):
    runner, policy = runtime
    with ThreadPoolExecutor(max_workers=1) as executor:
        consumer = executor.submit(current_thread).result(JOIN_TIMEOUT_SECONDS)
        result, expected, _, _ = make_step(runner, policy, consumer, 0)
        with pytest.raises(ValueError, match="different consumer"):
            result.get_output()
        assert not result.attempted
        assert executor.submit(result.get_output).result(JOIN_TIMEOUT_SECONDS).sampled_token_ids == expected


def test_optional_output_uses_original_copy_and_preserves_nan_counts(runtime):
    runner, policy = runtime
    result, expected, _, _ = make_step(runner, policy, current_thread(), 0, optional=True)
    assert type(result) is AsyncOutput
    output = result.get_output()
    assert output.sampled_token_ids == expected
    assert list(output.num_nans_in_logits.values()) == [0] * ROWS
    assert not policy.pending


def test_nonreply_returns_metadata_without_deferred_output(runtime):
    runner, policy = runtime
    result, _, _, _ = make_step(runner, policy, current_thread(), 0, reply=False)
    assert type(result) is ModelRunnerOutput
    assert result.req_ids == [f"request-0-{row}" for row in range(ROWS)]
    assert not policy.pending


def supported_worker(device):
    from vllm.models.deepseek_v41.nvidia.model_state import DeepseekV41ModelState
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    from vllm.v1.worker.gpu.sample.sampler import Sampler
    from vllm.v1.worker.gpu.spec_decode.adaptive_verification import AdaptiveVerificationManager
    from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator
    from vllm.v1.worker.gpu.spec_decode.rejection_sampler import RejectionSampler

    runner = object.__new__(GPUModelRunner)
    runner.device = device
    runner.scheduler_config = SimpleNamespace(async_scheduling=True)
    runner.max_model_len = MAX_CONTEXT
    runner.max_num_reqs = MAX_REQUESTS
    runner.model_config = SimpleNamespace(served_model_name="deepseek-v4-1-flash")
    runner.speculative_config = SimpleNamespace(
        method="dspark", num_speculative_tokens=DEPTH,
        num_speculative_tokens_per_batch_size=None, enable_adaptive_verification=True,
        draft_sample_method="probabilistic", rejection_sample_method="block")
    runner.num_speculative_steps = DEPTH
    runner.sampler = object.__new__(Sampler)
    runner.rejection_sampler = object.__new__(RejectionSampler)
    runner.rejection_sampler.sampler = runner.sampler
    runner.speculator = object.__new__(DSparkSpeculator)
    runner.speculator.acceptance_estimator = None
    runner.speculator.supports_mm_inputs = False
    runner.model_state = object.__new__(DeepseekV41ModelState)
    runner.adaptive_verification = object.__new__(AdaptiveVerificationManager)
    parallel = SimpleNamespace(tensor_parallel_size=TENSOR_PARALLEL_SIZE,
                               data_parallel_size=1, pipeline_parallel_size=1,
                               prefill_context_parallel_size=1,
                               decode_context_parallel_size=1, enable_expert_parallel=False)
    return SimpleNamespace(rank=0, device=device, model_runner=runner,
                           parallel_config=parallel)


@pytest.mark.parametrize("path,value", [
    ("parallel_config.tensor_parallel_size", 8),
    ("parallel_config.pipeline_parallel_size", 2),
    ("parallel_config.enable_expert_parallel", True),
    ("model_runner.scheduler_config.async_scheduling", False),
    ("model_runner.max_num_reqs", MAX_REQUESTS // 2),
    ("model_runner.max_model_len", MAX_CONTEXT // 2),
    ("model_runner.model_config.served_model_name", "gpt-oss-120b"),
    ("model_runner.speculative_config.method", "mtp"),
    ("model_runner.speculative_config.enable_adaptive_verification", False),
    ("model_runner.speculative_config.num_speculative_tokens", DEPTH - 1),
    ("model_runner.speculative_config.rejection_sample_method", "standard"),
    ("model_runner.speculator.supports_mm_inputs", True),
])
def test_config_guard_rejects_unsupported_runtime_variants(runtime, path, value):
    runner, _ = runtime
    worker = supported_worker(runner.device)
    assert _supported_config(worker)
    target = worker
    fields = path.split(".")
    for field in fields[:-1]:
        target = getattr(target, field)
    setattr(target, fields[-1], value)
    assert not _supported_config(worker)
