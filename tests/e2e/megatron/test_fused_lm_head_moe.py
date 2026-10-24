"""``--log-probs-backend fused_lm_head`` against ``fused`` on a small MoE model under rollout routing replay.

On eight GPUs, experts split over EP4 or EP8 with expert TP1 and full activation recompute, as in the
Qwen3.5 MoE recipes: the RL shape (TP4 + SP, CP1, PP1, EP8 over TP x DP), CP2 with PP2 (the SFT shape,
scaled down), TP2 + SP with CP2, and the RL shape with a trained MTP layer, whose router routes on
its own, under --overlap-grad-reduce. Each backend must replay every recorded routing once in the forward
and once in the recomputing backward and match the other's gradients, losses and scores
(``fused_lm_head_harness`` has the comparison).
"""

import sys

from tests.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=500, suite="stage-c-8-gpu-h200", labels=["megatron"], hardware=["hopper", "blackwell"], num_gpus=8
)

import pytest
from tests.e2e.megatron.fused_lm_head_harness import Layout, run_layout

_MOE = dict(world_size=8, num_experts=8, recompute=True)
_LAYOUTS = {
    "tp4_sp_ep8_thd": Layout(tp=4, sequence_parallel=True, cp=1, pp=1, qkv_format="thd", tied=False, ep=8, **_MOE),
    "tp2_sp_cp2_pp2_ep4_thd": Layout(
        tp=2, sequence_parallel=True, cp=2, pp=2, qkv_format="thd", tied=False, ep=4, **_MOE
    ),
    "tp2_sp_cp2_ep4_bshd": Layout(
        tp=2, sequence_parallel=True, cp=2, pp=1, qkv_format="bshd", tied=False, ep=4, **_MOE
    ),
    "tp4_sp_ep8_thd_mtp_overlap": Layout(
        tp=4,
        sequence_parallel=True,
        cp=1,
        pp=1,
        qkv_format="thd",
        tied=False,
        ep=8,
        mtp_layers=1,
        overlap_grad_reduce=True,
        **_MOE,
    ),
}


@pytest.mark.parametrize("layout", list(_LAYOUTS.values()), ids=list(_LAYOUTS))
def test_fused_lm_head_matches_fused_under_routing_replay(layout):
    run_layout(layout)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
