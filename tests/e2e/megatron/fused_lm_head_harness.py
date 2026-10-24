"""``--log-probs-backend fused_lm_head`` against ``fused`` through Megatron's pipeline schedule.

A small GPT model runs Miles' own training step (``run_forward_backward_pass``) and no-grad scoring
(``forward_only``) over one ``Layout``: TP with and without sequence parallelism, zigzag CP whose ranks
split the responses, PP, THD and BSHD, three accumulated microbatches per step, an untied or a tied output
layer, and optionally MoE layers split over expert-parallel ranks. Both backends start from the same
weights: every parameter's accumulated gradient, the losses and the scores must agree. A head-only test
cannot show this, because the hidden states' gradient has to travel back through the CP, SP, EP and
pipeline communication of the whole model.

MoE layouts train under rollout routing replay with full activation recompute, as the RL recipes do. The
routing is recorded under perturbed router weights, standing in for a rollout engine's, so it differs from
the trainer's own; each backend must consume every recorded routing exactly once in the forward and once
in the recomputing backward, and the replay must visibly move the gradient.

MTP layouts train MTP layers too: ``fused`` takes Megatron's own MTP loss on materialized logits, and
``fused_lm_head`` routes it through the tiles; the MTP losses Megatron logs must agree as well.
"""

import contextlib
import dataclasses
import faulthandler
import os
from argparse import Namespace
from unittest import mock

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from megatron.core import parallel_state as mpu
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig, finalize_model_grads
from megatron.core.models.gpt import GPTModel
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec, get_gpt_mtp_block_spec
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.module import Float16Module
from megatron.core.transformer.multi_token_prediction import MTPLossLoggingHelper
from megatron.training.global_vars import set_args
from tests.fast.backends.training_utils.loss.loss_test_utils import make_args
from tests.fast.dist_utils import find_free_port

from miles.backends.megatron_utils import model as megatron_model
from miles.backends.megatron_utils.lm_head import setup_lm_head
from miles.backends.megatron_utils.parallel import create_megatron_parallel_state
from miles.backends.training_utils.data.context_parallel import slice_log_prob_with_cp
from miles.backends.training_utils.data.rollout import DataIterator
from miles.backends.training_utils.loss.hub.logit_processors import get_log_probs_and_entropy
from miles.backends.training_utils.parallel import GroupInfo, get_parallel_state, set_parallel_state
from miles.utils.replay_base import routing_replay_manager

_VOCAB = 1000
_PADDED_VOCAB = 1024
_HIDDEN = 128
_NUM_MICROBATCHES = 3
_MICRO_BATCH_SIZE = 2
# (prompt, response) per sample; the long prompts put whole zigzag chunks of a sample inside its prompt
_SAMPLES = [(40, 9), (6, 25), (3, 14), (30, 2), (11, 17), (5, 31)]
_BSHD_SEQ_LEN = 64
_WATCHDOG_SECONDS = 600
_MOE_TOPK = 2
# std of the noise added to the router weights (themselves std 0.05) to record a rollout's routing
_ROLLOUT_ROUTER_NOISE = 0.05
_GRAD_TOLERANCE = 2e-2


@dataclasses.dataclass(frozen=True)
class Layout:
    tp: int
    sequence_parallel: bool
    cp: int
    pp: int
    qkv_format: str
    tied: bool
    world_size: int = 4
    num_experts: int = 0  # 0: dense MLPs
    ep: int = 1  # expert-parallel ranks; the expert tensor parallelism is 1, as in the recipes
    # full activation recompute: the backward reruns every layer's forward, MoE routers included
    recompute: bool = False
    mtp_layers: int = 0  # MTP depths trained alongside the policy (--enable-mtp-training)
    mtp_detach_heads: bool = True  # Miles' default: the MTP loss trains only the MTP layers
    # DDP reduces each bucket as soon as every parameter's gradient hook has fired, and asserts on one that never does
    overlap_grad_reduce: bool = False


LOSSES = {
    "sft": dict(loss_type="sft_loss", rollout_temperature=1.0),
    "cispo": dict(
        loss_type="policy_loss",
        policy_loss_type="cispo",
        eps_clip=0.2,
        eps_clip_high=0.28,
        entropy_coef=0.01,
        use_kl_loss=True,
        kl_loss_coef=0.05,
        kl_loss_type="low_var_kl",
        rollout_temperature=0.8,
    ),
}


def run_layout(layout: Layout) -> None:
    if torch.cuda.device_count() < layout.world_size:
        raise RuntimeError(f"requires {layout.world_size} GPUs, found {torch.cuda.device_count()}")
    mp.spawn(_worker, args=(find_free_port(), layout), nprocs=layout.world_size, join=True)


# ---------------------------------------------------------------------------
# World, model and data
# ---------------------------------------------------------------------------


def _worker(rank: int, port: int, layout: Layout) -> None:
    # a rank stuck in a collective fails the test with every thread's stack instead of hanging the job
    faulthandler.dump_traceback_later(_WATCHDOG_SECONDS, exit=True)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["NVTE_ALLOW_NONDETERMINISTIC_ALGO"] = "0"
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=layout.world_size)
    mpu.initialize_model_parallel(
        tensor_model_parallel_size=layout.tp,
        pipeline_model_parallel_size=layout.pp,
        context_parallel_size=layout.cp,
        expert_model_parallel_size=layout.ep,
        expert_tensor_parallel_size=1 if layout.num_experts else None,
    )
    model_parallel_cuda_manual_seed(1234)
    set_args(Namespace(cp_comm_type="p2p"))  # all create_megatron_parallel_state reads from Megatron's args
    set_parallel_state(create_megatron_parallel_state(indep_dp=GroupInfo(rank=0, size=1, group=None)))
    # the routers register their replay streams while the model is built
    routing_replay_manager.enabled = bool(layout.num_experts)
    _compare_backends(layout)
    print(f"rank {rank}: {layout} matches", flush=True)
    # no destroy_process_group: after the pipeline's P2P traffic it can block forever on this stack, and
    # the worker's exit releases the communicators anyway
    faulthandler.cancel_dump_traceback_later()


def _model(layout: Layout) -> DistributedDataParallel:
    moe = {}
    if layout.num_experts:
        # the shape of the Qwen3.5 MoE recipes' experts, scaled down
        moe = dict(
            num_moe_experts=layout.num_experts,
            moe_router_topk=_MOE_TOPK,
            moe_ffn_hidden_size=_HIDDEN,
            moe_shared_expert_intermediate_size=_HIDDEN,
            moe_shared_expert_gate=True,
            moe_router_score_function="softmax",
            moe_token_dispatcher_type="alltoall",
            moe_grouped_gemm=True,
            moe_permute_fusion=True,
            moe_router_dtype="fp32",
            moe_aux_loss_coeff=0.0,
            expert_model_parallel_size=layout.ep,
            expert_tensor_parallel_size=1,
        )
    if layout.recompute:
        moe.update(recompute_granularity="full", recompute_method="uniform", recompute_num_layers=1)
    if layout.mtp_layers:
        moe.update(
            mtp_num_layers=layout.mtp_layers, mtp_loss_scaling_factor=0.2, mtp_detach_heads=layout.mtp_detach_heads
        )
    config = TransformerConfig(
        num_layers=2 * layout.pp,
        hidden_size=_HIDDEN,
        num_attention_heads=4,
        ffn_hidden_size=2 * _HIDDEN,
        tensor_model_parallel_size=layout.tp,
        pipeline_model_parallel_size=layout.pp,
        context_parallel_size=layout.cp,
        sequence_parallel=layout.sequence_parallel,
        cp_comm_type="p2p",
        bf16=True,
        params_dtype=torch.bfloat16,
        pipeline_dtype=torch.bfloat16,
        variable_seq_lengths=True,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        normalization="RMSNorm",
        add_bias_linear=False,
        gated_linear_unit=True,
        activation_func=F.silu,
        gradient_accumulation_fusion=True,
        init_method_std=0.05,
        **moe,
    )
    layer_spec = get_gpt_layer_with_transformer_engine_spec(
        num_experts=layout.num_experts or None, moe_grouped_gemm=bool(layout.num_experts)
    )
    gpt = GPTModel(
        config=config,
        transformer_layer_spec=layer_spec,
        mtp_block_spec=(
            get_gpt_mtp_block_spec(config, layer_spec, use_transformer_engine=True) if layout.mtp_layers else None
        ),
        vocab_size=_PADDED_VOCAB,
        max_sequence_length=256,
        pre_process=get_parallel_state().pp.rank == 0,
        post_process=get_parallel_state().is_pp_last_stage,
        parallel_output=True,
        share_embeddings_and_output_weights=layout.tied,
        position_embedding_type="rope",
    ).cuda()
    model = DistributedDataParallel(
        config=config,
        ddp_config=DistributedDataParallelConfig(
            grad_reduce_in_fp32=True, overlap_grad_reduce=layout.overlap_grad_reduce
        ),
        module=Float16Module(config, gpt),
    )
    config.finalize_model_grads_func = finalize_model_grads
    # as Miles sets it: no reduction until the step's last microbatch
    config.no_sync_func = model.no_sync if layout.overlap_grad_reduce else None
    config.grad_scale_func = None
    config.timers = None
    return model


def _args(layout: Layout, backend: str, loss_overrides: dict) -> Namespace:
    return make_args(
        **{
            "true_on_policy_mode": False,
            "log_probs_backend": backend,
            "lm_head_logits_dtype": "model",
            "log_probs_chunk_size": 16,
            "vocab_size": _VOCAB,
            "qkv_format": layout.qkv_format,
            "data_pad_size_multiplier": 8,
            "allgather_cp": False,
            "observe_training_entropy": False,
            "use_rollout_entropy": True,
            "enable_mtp_training": False,
            "enable_witness": False,
            "custom_megatron_before_log_prob_hook_path": None,
            "use_dynamic_batch_size": False,
            "seq_length": 128,
            "micro_batch_size": _MICRO_BATCH_SIZE,
            "decoder_seq_length": None,
            "global_batch_size": len(_SAMPLES),
            **loss_overrides,
        }
    )


def _rollout_data(layout: Layout) -> dict:
    """Identical on every rank: the pipeline and CP ranks each slice what they hold."""
    gen = torch.Generator().manual_seed(29)
    data = {key: [] for key in ("tokens", "total_lengths", "response_lengths", "loss_masks", "advantages")}
    for prompt, response in _SAMPLES:
        tokens = torch.randint(0, _VOCAB, (prompt + response,), generator=gen)
        tokens[-1] = _VOCAB - 1  # the highest real id, next to the vocab padding
        mask = torch.ones(response)
        mask[1::4] = 0.0
        data["tokens"].append(tokens.cuda())
        data["total_lengths"].append(prompt + response)
        data["response_lengths"].append(response)
        data["loss_masks"].append(mask.cuda())
        data["advantages"].append(torch.randn(response, generator=gen).cuda())
    old = [(-3 + torch.randn(r, generator=gen)).cuda() for r in data["response_lengths"]]
    data["log_probs"] = old
    data["rollout_log_probs"] = [lp + 0.1 for lp in old]
    data["ref_log_probs"] = [lp - 0.1 for lp in old]
    if layout.qkv_format == "bshd":
        data["max_seq_lens"] = [_BSHD_SEQ_LEN] * len(_SAMPLES)
    # per-token scores and advantages arrive CP-local, as Miles' data pipeline hands them over; masks stay whole
    for key in ("log_probs", "rollout_log_probs", "ref_log_probs", "advantages"):
        data[key] = [
            slice_log_prob_with_cp(value, total, response, layout.qkv_format, _BSHD_SEQ_LEN)
            for value, total, response in zip(data[key], data["total_lengths"], data["response_lengths"], strict=True)
        ]
    return data


class _NoDumper:
    def __init__(self, *args, **kwargs):
        pass

    def wrap_forward_step(self, forward_step_func):
        return forward_step_func

    def finalize(self, model):
        pass


# ---------------------------------------------------------------------------
# Rollout routing replay
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _replay_stage(stage: str):
    """Every recorded routing from its first microbatch on, under ``stage``, as the actor sets it."""
    for replay in routing_replay_manager.replays:
        replay.forward_index = replay.backward_index = 0
    routing_replay_manager.stage = stage
    try:
        yield
    finally:
        routing_replay_manager.stage = "fallthrough"


def _record_rollout_routing(model: DistributedDataParallel, args: Namespace, data: dict) -> None:
    """Record a routing per MoE layer and microbatch that differs from the trainer's own.

    A rollout engine routes from its own copy of the weights; perturbed router weights stand in for it
    and are restored after recording.
    """
    routers = [param for name, param in model.module.named_parameters() if name.endswith("router.weight")]
    assert routers, "the MoE layout built no routers"
    saved = [param.detach().clone() for param in routers]
    gen = torch.Generator(device="cuda").manual_seed(31)  # one draw per router, the same on every rank
    with torch.no_grad():
        for param in routers:
            noise = torch.randn(param.shape, device=param.device, generator=gen) * _ROLLOUT_ROUTER_NOISE
            param.add_(noise.to(param.dtype))
    with _replay_stage("record"):
        _scores(model, args, data)
    with torch.no_grad():
        for param, value in zip(routers, saved, strict=True):
            param.copy_(value)
    lengths = {len(replay.top_indices_list) for replay in routing_replay_manager.replays}
    assert lengths == {_NUM_MICROBATCHES}, f"recorded {lengths} routings per router, not one per microbatch"


def _check_routing_consumed(*, backward: bool) -> None:
    """Each recorded routing was replayed once in the forward and, under recompute, once in the backward."""
    for index, replay in enumerate(routing_replay_manager.replays):
        recorded = len(replay.top_indices_list)
        assert (
            replay.forward_index == recorded
        ), f"router {index}: {replay.forward_index} forward replays of {recorded}"
        expected = recorded if backward else 0
        assert replay.backward_index == expected, f"router {index}: {replay.backward_index} backward replays"


# ---------------------------------------------------------------------------
# One backend's training step and scores
# ---------------------------------------------------------------------------


def _train_step(model: DistributedDataParallel, args: Namespace, data: dict, layout: Layout, *, replay: bool = True):
    model.zero_grad_buffer()
    iterator = DataIterator(data, micro_batch_size=_MICRO_BATCH_SIZE)
    replay = replay and bool(layout.num_experts)
    # the actor trains under replay_backward; the forward step switches to replay_forward around the model
    with _replay_stage("replay_backward") if replay else contextlib.nullcontext():
        losses = megatron_model.run_forward_backward_pass(
            args, _NoDumper(), [iterator], [model], _NUM_MICROBATCHES, num_rollouts=len(_SAMPLES)
        )
        if replay:
            _check_routing_consumed(backward=layout.recompute)
    grads = {name: param.main_grad.detach().clone() for name, param in model.module.named_parameters()}
    return losses, grads, _take_mtp_losses()


def _take_mtp_losses() -> torch.Tensor | None:
    """The per-depth MTP losses Megatron logged for this step on this rank, then cleared for the next."""
    tracker = MTPLossLoggingHelper.tracker
    if "loss_sums" not in tracker:
        return None
    losses = tracker["loss_sums"].clone()
    MTPLossLoggingHelper.clean_loss_in_tracker()
    return losses


def _scores(model: DistributedDataParallel, args: Namespace, data: dict) -> dict:
    iterator = DataIterator(data, micro_batch_size=_MICRO_BATCH_SIZE)
    with mock.patch.object(megatron_model, "DumperMegatronUtil", _NoDumper):
        return megatron_model.forward_only(
            get_log_probs_and_entropy, args, [model], [iterator], [_NUM_MICROBATCHES], rollout_id=0, fp32_output=False
        )


def _replayed_scores(model: DistributedDataParallel, args: Namespace, data: dict, layout: Layout) -> dict:
    if not layout.num_experts:
        return _scores(model, args, data)
    with _replay_stage("replay_forward"):
        scores = _scores(model, args, data)
        _check_routing_consumed(backward=False)
    return scores


def _relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    expected = expected.double()
    return ((actual.double() - expected).abs().max() / expected.abs().max().clamp_min(1e-12)).item()


def _compare_backends(layout: Layout) -> None:
    model = _model(layout)
    setup_lm_head(_args(layout, "fused_lm_head", LOSSES["sft"]), [model])
    data = _rollout_data(layout)
    if layout.num_experts:
        _record_rollout_routing(model, _args(layout, "fused", LOSSES["sft"]), data)
    for loss_name, loss_overrides in LOSSES.items():
        results = {}
        for backend in ("fused", "fused_lm_head"):
            args = _args(layout, backend, loss_overrides)
            results[backend] = (
                _train_step(model, args, data, layout),
                _replayed_scores(model, args, data, layout),
            )
        ((fused_losses, fused_grads, fused_mtp), fused_scores), ((lm_losses, lm_grads, lm_mtp), lm_scores) = (
            results["fused"],
            results["fused_lm_head"],
        )
        where = f"{loss_name} on rank {dist.get_rank()}"
        _check_mtp_losses(layout, lm_mtp, fused_mtp, where)
        assert (
            len(lm_losses) == len(fused_losses) == (_NUM_MICROBATCHES if get_parallel_state().is_pp_last_stage else 0)
        )
        for lm_loss, fused_loss in zip(lm_losses, fused_losses, strict=True):
            for key in fused_loss["keys"]:
                index = fused_loss["keys"].index(key) + 1
                torch.testing.assert_close(
                    lm_loss["values"][index], fused_loss["values"][index], rtol=2e-3, atol=2e-4, msg=f"{where}: {key}"
                )
        assert lm_grads.keys() == fused_grads.keys()
        unused = _unused_parameters(model)
        for name, fused_grad in fused_grads.items():
            # every parameter of every stage the forward reads takes part in both backends' backward
            assert name in unused or fused_grad.abs().max() > 0, f"{where}: {name} got no gradient"
            error = _relative_error(lm_grads[name], fused_grad)
            assert error < _GRAD_TOLERANCE, f"{where}: {name} gradient differs by {error:.3g} of its scale"
        if layout.num_experts:
            _check_replay_moves_the_gradient(model, _args(layout, "fused", loss_overrides), data, layout, fused_grads)
        assert lm_scores.keys() == fused_scores.keys()
        if get_parallel_state().is_pp_last_stage:
            assert set(lm_scores) == {"log_probs", "entropy"}
        # the same backend twice: how far the backbone alone moves the scores between two forwards
        control_scores = _replayed_scores(model, _args(layout, "fused", loss_overrides), data, layout)
        for key, fused_values in fused_scores.items():
            rows = [
                (value.numel(), _mean_abs(lm, value), _max_abs(lm, value), _mean_abs(control, value))
                for lm, value, control in zip(lm_scores[key], fused_values, control_scores[key], strict=True)
            ]
            print(f"{where} {key}: (rows, mean |lm - fused|, max |lm - fused|, mean |fused - fused|) {rows}")
            for lm_value, fused_value in zip(lm_scores[key], fused_values, strict=True):
                torch.testing.assert_close(lm_value, fused_value, rtol=0, atol=2e-2, msg=f"{where}: {key}")
                assert _mean_abs(lm_value, fused_value) < 1e-3, f"{where}: {key} {rows}"


def _unused_parameters(model: DistributedDataParallel) -> set[str]:
    """Parameters the forward never reads: a last stage's own output weight, when MTP keeps the tied
    embedding there and Megatron projects with the embedding's weight instead."""
    gpt = model.module.module
    layer_weight = gpt.output_layer.weight if gpt.post_process else None
    if layer_weight is None or not gpt.share_embeddings_and_output_weights:
        return set()
    return set() if gpt.shared_embedding_or_output_weight() is layer_weight else {"module.output_layer.weight"}


def _check_replay_moves_the_gradient(
    model: DistributedDataParallel, args: Namespace, data: dict, layout: Layout, replayed_grads: dict
) -> None:
    """Without the replay the trainer routes on its own; were the gradients equal, the comparison proved nothing."""
    routing_replay_manager.enabled = False  # the routers read it per call
    try:
        _, own_grads, _ = _train_step(model, args, data, layout, replay=False)
    finally:
        routing_replay_manager.enabled = True
    errors = {name: _relative_error(own_grads[name], grad) for name, grad in replayed_grads.items()}
    name = max(errors, key=errors.get)
    print(f"rank {dist.get_rank()}: own routing moves the gradient by up to {errors[name]:.3g} of its scale ({name})")
    assert errors[name] > 5 * _GRAD_TOLERANCE, f"the replayed routing changed no gradient beyond {errors[name]:.3g}"


def _check_mtp_losses(layout: Layout, lm_mtp: torch.Tensor | None, fused_mtp: torch.Tensor | None, where: str) -> None:
    """Megatron logs the MTP loss on the last stage; the tiled one must log what its materialized one does."""
    if not (layout.mtp_layers and get_parallel_state().is_pp_last_stage):
        assert lm_mtp is None and fused_mtp is None, f"{where}: MTP losses without MTP layers"
        return
    assert lm_mtp is not None and fused_mtp is not None, f"{where}: no MTP loss logged"
    print(f"{where} MTP loss per depth: fused_lm_head {lm_mtp.tolist()}, fused {fused_mtp.tolist()}")
    assert (fused_mtp > 0).all(), f"{where}: {fused_mtp}"
    torch.testing.assert_close(lm_mtp, fused_mtp, rtol=1e-4, atol=1e-5, msg=f"{where}: MTP loss")


def _mean_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a - b).abs().mean().item() if a.numel() else 0.0


def _max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a - b).abs().max().item() if a.numel() else 0.0
