"""``--log-probs-backend fused_lm_head`` against ``fused`` through Megatron's pipeline schedule, dense model.

On four GPUs: TP2 with and without sequence parallelism, zigzag CP2 whose ranks split the responses, PP2,
THD and BSHD, an untied and a tied output layer, MTP layers trained alongside the policy with and
without detached heads, and DDP reducing gradients as their hooks fire (``--overlap-grad-reduce``);
``fused_lm_head_harness`` has the comparison.
"""

import sys

from tests.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=600, suite="stage-c-4-gpu-h200", labels=["megatron"], hardware=["hopper", "blackwell"], num_gpus=4
)

import pytest
from tests.e2e.megatron.fused_lm_head_harness import Layout, run_layout

_LAYOUTS = {
    "tp2_sp_cp2_thd": Layout(tp=2, sequence_parallel=True, cp=2, pp=1, qkv_format="thd", tied=False),
    "tp2_sp_pp2_thd_tied": Layout(tp=2, sequence_parallel=True, cp=1, pp=2, qkv_format="thd", tied=True),
    "cp2_pp2_thd": Layout(tp=1, sequence_parallel=False, cp=2, pp=2, qkv_format="thd", tied=False),
    "tp2_sp_pp2_bshd_overlap": Layout(
        tp=2, sequence_parallel=True, cp=1, pp=2, qkv_format="bshd", tied=False, overlap_grad_reduce=True
    ),
    "tp2_cp2_bshd": Layout(tp=2, sequence_parallel=False, cp=2, pp=1, qkv_format="bshd", tied=False),
    "tp2_sp_cp2_thd_mtp_overlap": Layout(
        tp=2,
        sequence_parallel=True,
        cp=2,
        pp=1,
        qkv_format="thd",
        tied=False,
        mtp_layers=1,
        overlap_grad_reduce=True,
    ),
    # the MTP loss also trains the backbone and the tied output weight, across the pipeline's embedding copies
    "tp2_sp_pp2_thd_tied_mtp2_undetached": Layout(
        tp=2,
        sequence_parallel=True,
        cp=1,
        pp=2,
        qkv_format="thd",
        tied=True,
        mtp_layers=2,
        mtp_detach_heads=False,
        # not overlap_grad_reduce: Megatron leaves this last stage's own output weight unused (the MTP
        # embedding's weight is passed instead), and DDP rejects a parameter whose gradient hook never fires
    ),
    "tp2_cp2_bshd_mtp": Layout(tp=2, sequence_parallel=False, cp=2, pp=1, qkv_format="bshd", tied=False, mtp_layers=1),
}


@pytest.mark.parametrize("layout", list(_LAYOUTS.values()), ids=list(_LAYOUTS))
def test_fused_lm_head_matches_fused_through_the_schedule(layout):
    run_layout(layout)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
