"""Check the measured shape and hardware boundaries for expert selection."""

from types import SimpleNamespace

import pytest
import torch

from flashinfer.fused_moe import cc_deepseek_tactics as policy

ORIGINAL_TACTIC = (32, 527)
DRAFT_EXPERTS = 128
DRAFT_TOP_K = 3
UNMEASURED_ROWS = (1, 5, 7, 30, 36, 42, 48)


def fixture(rows):
    runner = SimpleNamespace(
        top_k=policy.TOP_K, num_experts=policy.EXPERTS, num_local_experts=policy.EXPERTS,
        hidden_size=policy.HIDDEN_SIZE, hidden_size_output=policy.HIDDEN_SIZE,
        intermediate_size=policy.INTERMEDIATE_SIZE, dtype_act=policy.DtypeTrtllmGen.MxE4m3,
        dtype_weights=policy.DtypeTrtllmGen.MxE2m1, activation_type=policy.ActivationType.Swiglu,
        weight_layout=policy.WeightLayout.MajorK, use_shuffled_weight=True,
        use_per_token_scaling=False, num_fused_shared_experts=0,
    )
    hidden = torch.empty((rows, policy.HIDDEN_SIZE), device="cuda", dtype=torch.bfloat16)
    return runner, SimpleNamespace(hidden_states=hidden, output=hidden, gemm1_lora_delta=None)


def select(runner, inputs, **kwargs):
    return policy.select_target_tactic(
        ORIGINAL_TACTIC, runner, inputs,
        do_finalize=kwargs.get("do_finalize", True),
        routing_input_mode=policy.RoutingInputMode.UnpackedPrecomputed,
        tuning=kwargs.get("tuning", False),
    )


@pytest.mark.parametrize("rows", sorted(policy.TARGET_ROWS) + list(UNMEASURED_ROWS))
def test_only_measured_target_rows_change(rows):
    assert policy.ENABLED
    runner, inputs = fixture(rows)
    expected = policy.TARGET_TACTIC if rows in policy.TARGET_ROWS else ORIGINAL_TACTIC
    assert select(runner, inputs) == expected
    assert select(runner, inputs, tuning=True) == ORIGINAL_TACTIC
    assert select(runner, inputs, do_finalize=False) == ORIGINAL_TACTIC
    runner.num_experts = runner.num_local_experts = DRAFT_EXPERTS
    runner.top_k = DRAFT_TOP_K
    assert select(runner, inputs) == ORIGINAL_TACTIC


def test_cpu_and_output_dtype_are_excluded():
    runner, inputs = fixture(policy.TARGET_ROWS_PER_REQUEST)
    inputs.output = inputs.output.float()
    assert select(runner, inputs) == ORIGINAL_TACTIC
    inputs.output = inputs.hidden_states
    inputs.hidden_states = inputs.hidden_states.cpu()
    assert select(runner, inputs) == ORIGINAL_TACTIC
