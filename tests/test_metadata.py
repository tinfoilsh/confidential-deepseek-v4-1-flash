"""Exercise actual CUDA metadata publication."""

import os

import numpy as np
import pytest
import torch

from vllm.v1.worker.gpu.buffer_utils import CpuMetadataTensor

ROWS = 8
COLUMNS = 3
DTYPE = torch.int32
PARTIAL_ROWS = 3


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


def test_metadata_republishes_mutated_rows_after_partial_and_empty_views():
    metadata = CpuMetadataTensor((ROWS, COLUMNS), dtype=DTYPE)
    metadata.np[:] = np.arange(ROWS * COLUMNS).reshape(ROWS, COLUMNS)
    full = metadata.copy_to_uva()
    assert_published(full, metadata.np.copy())
    assert metadata.copy_to_uva() is full
    expected = metadata.np.copy()
    expected[-1] *= -1
    metadata.np[:] = expected
    assert_published(metadata.copy_to_uva(), expected)
    assert_published(metadata.copy_to_uva(PARTIAL_ROWS), expected[:PARTIAL_ROWS])
    assert_published(metadata.copy_to_uva(0), expected[:0])
    assert_published(metadata.copy_to_uva(), expected)
