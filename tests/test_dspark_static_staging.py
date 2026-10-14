"""Exercise static DSpark eligibility with real transfer and output lifetimes."""

from concurrent.futures import ThreadPoolExecutor
from threading import current_thread
from types import SimpleNamespace

import numpy as np
import torch

from test_dspark_input_staging import MAX_REQUESTS, OFFSET_SENTINEL, device, text_runtime
from test_dspark_readback import JOIN_TIMEOUT_SECONDS, STEPS, make_step, runtime, supported_worker
from vllm.v1.worker.gpu.cc_input_staging import private_tensor_h2d, supports_text_staging
from vllm.v1.worker.gpu.cc_readback import CCReadbackPolicy, _DeferredOutput


def test_static_reply_selection_preserves_pending_outputs(runtime):
    streams, _ = runtime
    worker = supported_worker(streams.device)
    runner = worker.model_runner
    runner.main_stream = streams.main_stream
    runner.output_copy_stream = streams.output_copy_stream
    runner.speculative_config.enable_adaptive_verification = False
    runner.adaptive_verification = None
    policy = CCReadbackPolicy(runner)
    with ThreadPoolExecutor(max_workers=1) as executor:
        consumer = executor.submit(current_thread).result(JOIN_TIMEOUT_SECONDS)
        process = SimpleNamespace(worker=SimpleNamespace(worker=worker), rank=worker.rank,
                                  use_async_scheduling=True, async_output_copy_thread=consumer)
        token = policy.select(process, ("sample_tokens", (), {}, worker.rank))
        try:
            assert policy.context.get() == (True, consumer)
            pending = [make_step(runner, policy, consumer, step) for step in range(STEPS)]
            for result, expected, history, lengths in pending:
                assert type(result) is _DeferredOutput
                output = executor.submit(result.get_output).result(JOIN_TIMEOUT_SECONDS)
                assert output.sampled_token_ids == expected
                assert lengths.cpu().tolist() == [len(tokens) for tokens in expected]
                for row, tokens in enumerate(expected):
                    assert history[row, :len(tokens)].cpu().tolist() == tokens
        finally:
            policy.reset(token)
    assert not policy.pending


def test_static_text_selection_preserves_mutated_scheduler_values(device):
    runner, scheduler, batch = text_runtime(device)
    runner.speculative_config.enable_adaptive_verification = False
    runner.adaptive_verification = None
    assert supports_text_staging(runner, scheduler, batch, 0)
    source = np.arange(MAX_REQUESTS, dtype=np.intp)[::-1].copy()
    expected = torch.tensor(source, dtype=torch.int32)
    result = private_tensor_h2d(source, device=device, dtype=torch.int32)
    source[:] = OFFSET_SENTINEL
    torch.testing.assert_close(result.cpu(), expected, rtol=0, atol=0)
    runner.speculative_config.enable_adaptive_verification = True
    assert not supports_text_staging(runner, scheduler, batch, 0)
