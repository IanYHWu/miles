"""Teacher-forced, token-weighted validation on the resident Megatron model."""

import torch
import torch.distributed as dist


def _masked_totals(scores, masks, *, contributes, device):
    totals = torch.zeros(2, dtype=torch.float64, device=device)
    if contributes:
        for score, mask in zip(scores, masks, strict=True):
            if score.shape != mask.shape:
                raise ValueError("Validation score/mask shapes differ")
            active = mask != 0
            totals[0] -= (score[active].double() * mask[active]).sum()
            totals[1] += mask.sum()
    return totals


def evaluate_sft(actor, rollout_id, rollout_data_ref) -> dict:
    """Return summed NLL/count; padding rows have zero masks, including the final batch.

    Only TP rank zero on the last pipeline stage contributes; CP and DP contribute
    disjoint tokens. The world reduction broadcasts the same result to every rank.
    """
    # Defer optional Megatron/CUDA imports so reduction helpers remain CPU-testable.
    from megatron.core.utils import get_model_config

    from miles.backends.megatron_utils.initialize import RandomState
    from miles.backends.megatron_utils.model import forward_only
    from miles.backends.training_utils.data.context_parallel import get_local_response_loss_masks
    from miles.backends.training_utils.data.rollout import get_data_iterator, get_rollout_data
    from miles.backends.training_utils.loss.hub.logit_processors import get_log_probs_and_entropy
    from miles.backends.training_utils.parallel import get_parallel_state

    if actor.args.loss_type != "sft_loss" or actor.args.offload_train:
        raise ValueError("SFT validation requires resident SFT training models")
    actor._heartbeat.bump()
    rng = RandomState.capture()
    training_modes = [(module, module.training) for model in actor.model for module in model.modules()]
    configs = {id(config): config for config in map(get_model_config, actor.model)}
    timers = [(config, config.timers) for config in configs.values()]
    try:
        data, lease = get_rollout_data(actor.args, rollout_data_ref)
        with lease, torch.no_grad():
            iterator, microbatches = get_data_iterator(actor.args, actor.model, data)
            output = forward_only(
                get_log_probs_and_entropy,
                actor.args,
                actor.model,
                iterator,
                microbatches,
                rollout_id,
                store_prefix="sft_val_",
                fp32_output=False,
            )
            parallel = get_parallel_state()
            contributes = parallel.is_pp_last_stage and parallel.tp.rank == 0
            masks = []
            if contributes:
                masks = get_local_response_loss_masks(
                    data["total_lengths"],
                    data["response_lengths"],
                    data["loss_masks"],
                    actor.args.qkv_format,
                    data.get("max_seq_lens"),
                )
            totals = _masked_totals(
                output.get("sft_val_log_probs", []),
                masks,
                contributes=contributes,
                device=torch.cuda.current_device(),
            )
            dist.all_reduce(totals)
            return dict(nll_sum=totals[0].item(), token_count=totals[1].item())
    finally:
        rng.restore()
        for module, training in training_modes:
            module.training = training
        for config, timer in timers:
            config.timers = timer
        actor._heartbeat.bump()
