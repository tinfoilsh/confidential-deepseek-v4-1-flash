"""Additional explicit-copy lifetime checks; run beside the image's UVA suite."""

import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from vllm.utils.platform_utils import is_uva_available
from vllm.v1.worker.gpu import buffer_utils


POOL_DEPTH = 2
INITIAL_ROWS = 2
TRAILING_COLUMNS = 3
POOL_SHAPES = ((INITIAL_ROWS,), (INITIAL_ROWS, TRAILING_COLUMNS))
POOL_LENGTHS = (1, 1, 257, 257, 3, 3, 4097, 4097, 2, 2)
OUT_LENGTHS = (0, 0, *POOL_LENGTHS)
GENERATION_STRIDE = 65536
MUTATED_VALUE = -1
DIRECT_ROWS = 8
DIRECT_PREFIX_ROWS = 3
DIRECT_LENGTHS = (None, None, DIRECT_PREFIX_ROWS, DIRECT_PREFIX_ROWS, 0, 0)
DTYPE = torch.int32
POLICY_ENV = "DEEPSEEK_CC_TRANSFER_POLICY"
POLICY_PINNING = {"A": False, "B": False}
CUDA_UNREGISTERED = 0
CUDA_HOST = 1
CUDA_DEVICE = 2
CUDA_MANAGED = 3
CONTROL_ELEMENTS = 8


def allocation_observation(tensor):
    from cuda.bindings import runtime

    error, attributes = runtime.cudaPointerGetAttributes(tensor.data_ptr())
    assert error == runtime.cudaError_t.cudaSuccess, str(error)
    memory_type = int(attributes.type)
    pinned = tensor.is_pinned()
    assert pinned == (memory_type == CUDA_HOST)
    return {"cuda_return_code": int(error), "cuda_memory_type": memory_type, "pinned": pinned}


@pytest.fixture(autouse=True)
def transfer_runtime(record_property):
    policy = os.environ.get(POLICY_ENV)
    assert policy in POLICY_PINNING, f"set {POLICY_ENV} to inspected A/B; C requires a revised allocation contract"
    assert os.environ.get("VLLM_DISABLE_UVA") == "1"
    assert torch.cuda.is_available(), "these cases require an actual CUDA execution"
    assert torch.cuda.device_count() == 1, "expose one GPU in the isolated diagnostic"
    assert not is_uva_available(), "the selected runtime must use explicit copies"
    controls = {
        "pageable": torch.empty(CONTROL_ELEMENTS, dtype=DTYPE, device="cpu", pin_memory=False),
        "requested_host": torch.empty(CONTROL_ELEMENTS, dtype=DTYPE, device="cpu", pin_memory=True),
        "device": torch.empty(CONTROL_ELEMENTS, dtype=DTYPE, device="cuda:0"),
    }
    control_observations = {name: allocation_observation(tensor) for name, tensor in controls.items()}
    for name, expected in {"pageable": CUDA_UNREGISTERED, "requested_host": CUDA_MANAGED, "device": CUDA_DEVICE}.items():
        assert control_observations[name]["cuda_memory_type"] == expected
    record_property("allocation_controls", json.dumps(control_observations, sort_keys=True))
    source = Path(buffer_utils.__file__)
    record_property("runtime", json.dumps({
        "policy": policy,
        "expected_slot_pinned": POLICY_PINNING[policy],
        "expected_slot_memory_type": CUDA_UNREGISTERED,
        "packages": {name: metadata.version(name) for name in ("vllm", "torch", "numpy")},
        "buffer_utils_path": str(source),
        "buffer_utils_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_capability": torch.cuda.get_device_capability(0),
    }, sort_keys=True))
    audit = {"pinned": POLICY_PINNING[policy], "buffers": [], "callers": []}
    with torch.cuda.device(0):
        yield audit
    record_property("slot_observations", json.dumps(audit["buffers"], sort_keys=True))
    record_property("caller_allocations", json.dumps(audit["callers"], sort_keys=True))


def observe_buffer(audit, label, buffer):
    observed = {
        "label": label, "shape": list(buffer.cpu.shape),
        **allocation_observation(buffer.cpu),
        "buffer_class": type(buffer).__name__,
    }
    audit["buffers"].append(observed)
    assert isinstance(buffer, buffer_utils.NonUvaBuffer)
    assert observed["pinned"] == audit["pinned"]
    assert observed["cuda_memory_type"] == CUDA_UNREGISTERED


def generation_values(shape, generation):
    return torch.arange(math.prod(shape), dtype=DTYPE, device="cpu").reshape(shape) + generation * GENERATION_STRIDE


def mutable_caller(reference, kind, audit):
    if kind == "list":
        return reference.tolist()
    if kind == "numpy":
        return reference.numpy().copy()
    if kind == "tensor":
        return reference.clone()
    if kind == "pinned":
        caller = torch.empty(reference.shape, dtype=DTYPE, device="cpu", pin_memory=True)
        caller.copy_(reference)
        observation = allocation_observation(caller)
        audit["callers"].append({"requested_pin_memory": True, **observation})
        assert observation["cuda_memory_type"] == CUDA_MANAGED
        return caller
    raise ValueError(f"unknown caller kind {kind}")


def mutate_caller(caller):
    if isinstance(caller, list):
        caller.clear()
    elif isinstance(caller, np.ndarray):
        caller.fill(MUTATED_VALUE)
    else:
        caller.fill_(MUTATED_VALUE)


def retire(pending, slot):
    item = pending.pop(slot, None)
    if item is not None:
        event, snapshot, reference = item
        event.synchronize()
        torch.testing.assert_close(snapshot.cpu(), reference, rtol=0, atol=0)


def exercise_pool(audit, input_kind, initial_shape, *, use_out=False, cross_stream=False):
    pool = buffer_utils.UvaBufferPool(initial_shape, DTYPE, max_concurrency=POOL_DEPTH)
    for slot, buffer in enumerate(pool._uva_bufs):
        observe_buffer(audit, f"initial-slot-{slot}", buffer)
    producer = torch.cuda.Stream()
    producer.wait_stream(torch.cuda.current_stream())
    consumer = torch.cuda.Stream() if cross_stream else producer
    pending = {}
    lengths = OUT_LENGTHS if use_out else POOL_LENGTHS
    for generation, length in enumerate(lengths):
        slot = (pool._curr + 1) % POOL_DEPTH
        retire(pending, slot)
        reference = generation_values((length, *initial_shape[1:]), generation)
        caller = mutable_caller(reference, input_kind, audit)
        with torch.cuda.stream(producer):
            if use_out:
                out = torch.empty(reference.shape, dtype=DTYPE, device="cuda:0")
                result = pool.copy_to_gpu(caller, out=out)
                assert result is out
            else:
                result = pool.copy_to_uva(caller)
            mutate_caller(caller)
            assert pool._curr == slot
            assert result.device == torch.device("cuda:0")
            assert tuple(result.shape) == tuple(reference.shape)
            observe_buffer(audit, f"generation-{generation}-slot-{slot}", pool._uva_bufs[slot])
            ready = torch.cuda.Event()
            ready.record(producer)
        with torch.cuda.stream(consumer):
            if cross_stream:
                consumer.wait_event(ready)
                result.record_stream(consumer)
            snapshot = result.clone()
            done = torch.cuda.Event()
            done.record(consumer)
        pending[slot] = (done, snapshot, reference)
    for slot in tuple(pending):
        retire(pending, slot)


@pytest.mark.parametrize("input_kind", ["list", "numpy", "tensor", "pinned"])
@pytest.mark.parametrize("initial_shape", POOL_SHAPES)
def test_pool_preserves_mutated_callers_through_growth_and_shorter_reuse(transfer_runtime, input_kind, initial_shape):
    exercise_pool(transfer_runtime, input_kind, initial_shape)


@pytest.mark.parametrize("input_kind", ["numpy", "tensor"])
def test_copy_to_gpu_out_preserves_generations_and_empty_shapes(transfer_runtime, input_kind):
    exercise_pool(transfer_runtime, input_kind, POOL_SHAPES[1], use_out=True)


def test_second_stream_consumes_after_producer_and_retires_before_reuse(transfer_runtime):
    exercise_pool(transfer_runtime, "pinned", POOL_SHAPES[1], cross_stream=True)


@pytest.mark.parametrize("shape", [(DIRECT_ROWS,), (DIRECT_ROWS, TRAILING_COLUMNS)])
def test_direct_full_prefix_and_empty_copy_preserve_retired_generations(transfer_runtime, shape):
    buffers = [buffer_utils.NonUvaBuffer(shape, DTYPE) for _ in range(POOL_DEPTH)]
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    pending = {}
    for generation, length in enumerate(DIRECT_LENGTHS):
        slot = generation % POOL_DEPTH
        retire(pending, slot)
        buffer = buffers[slot]
        observe_buffer(transfer_runtime, f"generation-{generation}-slot-{slot}", buffer)
        source = generation_values(shape, generation)
        buffer.cpu.copy_(source)
        reference = source.clone() if length is None else source[:length].clone()
        with torch.cuda.stream(stream):
            result = buffer.uva(length)
            assert result.device == torch.device("cuda:0")
            assert tuple(result.shape) == tuple(reference.shape)
            snapshot = result.clone()
            done = torch.cuda.Event()
            done.record(stream)
        pending[slot] = (done, snapshot, reference)
    for slot in tuple(pending):
        retire(pending, slot)
