"""Check sampling refresh and row reuse without later metadata patches."""

import os

import pytest
import torch

from vllm.sampling_params import SamplingParams
from vllm.v1.worker.gpu.sample.states import SamplingStates

ROWS = 8
VOCABULARY_SIZE = 256
INITIAL_SEED = 42
REUSED_SEED = 91
SAMPLING_FIELDS = ("temperature", "top_p", "top_k", "min_p", "seeds")
INITIAL_SAMPLING = {"temperature": 0.7, "top_p": 0.8, "top_k": 20,
                    "min_p": 0.1, "seed": INITIAL_SEED}
REUSED_SAMPLING = {"temperature": 0, "seed": REUSED_SEED}
TRAILING_SAMPLING = {"temperature": 1, "seed": INITIAL_SEED}


@pytest.fixture(autouse=True)
def explicit_copy_runtime():
    assert os.environ.get("VLLM_DISABLE_UVA") == "1"
    assert os.environ.get("VLLM_TINFOIL_CC_OPTIMIZATIONS") == "1"
    assert torch.cuda.is_available()
    assert torch.cuda.device_count() == 1
    with torch.cuda.device(0):
        yield
    torch.cuda.synchronize()


def assert_published(tensor, expected):
    torch.testing.assert_close(tensor.cpu(), torch.as_tensor(expected), rtol=0, atol=0)


def test_sampling_reused_row_publishes_new_values_without_changing_clean_rows():
    states = SamplingStates(ROWS, VOCABULARY_SIZE)
    for row in range(ROWS):
        states.add_request(row, SamplingParams(**TRAILING_SAMPLING))
    states.add_request(0, SamplingParams(**INITIAL_SAMPLING))
    states.apply_staged_writes()
    published = {name: getattr(states, name).gpu for name in SAMPLING_FIELDS}
    for name in SAMPLING_FIELDS:
        mirror = getattr(states, name)
        assert_published(mirror.gpu, mirror.np.copy())
    states.apply_staged_writes()
    for name in SAMPLING_FIELDS:
        assert getattr(states, name).gpu is published[name]
    states.add_request(0, SamplingParams(**REUSED_SAMPLING))
    states.add_request(ROWS - 1, SamplingParams(**TRAILING_SAMPLING))
    states.apply_staged_writes()
    for name in SAMPLING_FIELDS:
        mirror = getattr(states, name)
        assert_published(mirror.gpu, mirror.np.copy())
    assert states.seeds.gpu[0].item() == REUSED_SEED
    assert states.temperature.gpu[0].item() == 0
