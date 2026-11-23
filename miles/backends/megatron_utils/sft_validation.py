"""Teacher-forced, token-weighted validation on the resident Megatron model."""

import json
from pathlib import Path

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


def _read_probe_rows(data_path: str) -> list[dict]:
    """JSONL rows of ``tokens`` and a 0/1 ``loss_mask`` of the same length, ending in one contiguous target."""
    rows = [json.loads(line) for line in Path(data_path).read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"empty probe set {data_path}")
    for row in rows:
        mask = row["loss_mask"]
        if len(row["tokens"]) != len(mask) or not any(mask) or not set(mask) <= {0, 1}:
            raise ValueError("invalid probe token mask")
        if any(value != 1 for value in mask[mask.index(1) :]):
            raise ValueError("probe rows need a contiguous target")
    return rows


def probe_token_logprobs(args, model, *, data_path: str) -> list[list[float]] | None:
    """Teacher-forced log-probs of every target token of a fixed probe set, without changing weights.

    The same forward pass as training on a fixed batch, so two calls on the same weights give the same
    numbers: comparing calls measures how far an update moved the model. Needs CP = PP = 1 and a row
    count divisible by the DP size. Returns one list per row on global rank 0, ``None`` elsewhere."""
    from miles.backends.megatron_utils.model import forward_only
    from miles.backends.training_utils.data.rollout import get_data_iterator
    from miles.backends.training_utils.loss.hub.logit_processors import get_log_probs_and_entropy
    from miles.backends.training_utils.parallel import get_parallel_state

    state = get_parallel_state()
    if state.cp.size != 1 or args.pipeline_model_parallel_size != 1:
        raise ValueError("probe log-probs require CP=PP=1")
    rows = _read_probe_rows(data_path)
    dp = state.effective_dp.size
    if len(rows) % dp:
        raise ValueError(f"{len(rows)} probe rows do not split over DP={dp}")
    mine = list(range(state.effective_dp.rank, len(rows), dp))
    local = [rows[i] for i in mine]
    device = torch.cuda.current_device()
    response_lengths = [len(row["loss_mask"]) - row["loss_mask"].index(1) for row in local]
    batch = {
        "tokens": [torch.tensor(row["tokens"], dtype=torch.long, device=device) for row in local],
        "loss_masks": [
            torch.tensor(row["loss_mask"][-n:], dtype=torch.int, device=device)
            for row, n in zip(local, response_lengths, strict=True)
        ],
        "total_lengths": [len(row["tokens"]) for row in local],
        "response_lengths": response_lengths,
    }
    if args.use_dynamic_global_batch_size:
        batch["dynamic_global_batch_size"] = len(rows)  # one pass over the whole probe set
    iterators, microbatches = get_data_iterator(args, model, batch)
    with torch.no_grad():
        result = forward_only(
            get_log_probs_and_entropy,
            args,
            model,
            iterators,
            microbatches,
            rollout_id=0,
            store_prefix="probe_",
            fp32_output=False,
        )
    values = [[float(x) for x in torch.as_tensor(v).float().tolist()] for v in result["probe_log_probs"]]
    gathered = [None] * dp
    dist.all_gather_object(gathered, list(zip(mine, values, strict=True)), group=state.effective_dp.group)
    if dist.get_rank() != 0:
        return None
    by_row = dict(pair for part in gathered for pair in part)
    return [by_row[i] for i in range(len(rows))]
