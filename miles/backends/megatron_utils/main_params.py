"""Persist the distributed optimizer's fp32 main parameters across jobs, without Adam moments.

A job that takes only a few small optimizer steps loses most of each update when the next job
re-initialises main params from the bf16 weights, while a full optimizer checkpoint (fp32 params
plus two fp32 moments, ~14 bytes/param) can be too large to keep per run. These helpers save only
the fp32 main params (4 bytes/param), one set of safetensors files per (tp, pp) rank, holding each
local (TP-sharded) parameter in full. Loading needs the same TP/PP layout; DP and CP may differ because
every rank re-slices its own shard ranges.

Saving: every rank of a data-parallel group contributes the slices it owns; group rank 0 receives
the full tensor (``dist.reduce``) and writes. Loading: each rank reads only the rows covering its
own slices into ``shard_fp32_from_float16_groups`` / ``shard_fp32_groups`` and then checks that the
bf16 model params equal bf16(main) — they must, when the model was loaded from the HF export made
from the same main params at the end of the previous job.

Requires the Megatron ``DistributedOptimizer`` with main params on GPU (no precision-aware or CPU
offloaded optimizer).
"""

import json
import logging
import math
from pathlib import Path

import torch
import torch.distributed as dist

from miles.backends.training_utils.parallel import get_parallel_state

logger = logging.getLogger(__name__)

FORMAT = "miles-main-params/v1"
_PART_BYTES = 8 * 1024**3


def _optimizers(optimizer):
    opts = [
        o for o in getattr(optimizer, "chained_optimizers", [optimizer]) if not getattr(o, "is_stub_optimizer", False)
    ]
    for opt in opts:
        if not hasattr(opt, "shard_fp32_from_float16_groups"):
            raise TypeError(f"main-param persistence needs Megatron's DistributedOptimizer, got {type(opt).__name__}")
        if getattr(opt.config, "use_precision_aware_optimizer", False):
            raise TypeError("main-param persistence does not support --use-precision-aware-optimizer")
        if getattr(opt.config, "optimizer_cpu_offload", False):
            raise TypeError("main-param persistence does not support --optimizer-cpu-offload")
    return opts


def _owned(optimizer):
    """model param -> (shard main param, param range) for the slices this rank's optimizer owns."""
    owned, groups = {}, []
    for opt in _optimizers(optimizer):
        groups.append(opt.data_parallel_group)
        for model_groups, shard_groups in (
            (opt.model_float16_groups, opt.shard_fp32_from_float16_groups),
            (opt.model_fp32_groups, opt.shard_fp32_groups),
        ):
            for model_group, shard_group in zip(model_groups, shard_groups, strict=True):
                for model_param, shard_param in zip(model_group, shard_group, strict=True):
                    owned[model_param] = (shard_param, opt._get_model_param_range_map(model_param)["param"])
    ranks = {tuple(dist.get_process_group_ranks(g)) for g in groups}
    if len(ranks) != 1:
        raise RuntimeError(f"optimizers use different data-parallel groups: {ranks}")
    return owned, groups[0]


def _named_params(model):
    """(key, param) in a rank-independent order; keys drop DDP/Float16Module 'module.' prefixes."""
    for chunk_index, chunk in enumerate(model):
        for name, param in chunk.named_parameters():
            while name.startswith("module."):
                name = name[len("module.") :]
            yield f"{chunk_index}.{name}", param


def _file_stem():
    parallel = get_parallel_state()
    return f"tp{parallel.tp.rank:02d}-pp{parallel.pp.rank:02d}"


def save_main_params(model, optimizer, path: str) -> dict:
    """Collectively write fp32 main params under ``path``; returns a summary (same on all ranks)."""
    from safetensors.torch import save_file

    path = Path(path)
    owned, group = _owned(optimizer)
    group_rank = dist.get_rank(group)
    writer = group_rank == 0
    dst = dist.get_global_rank(group, 0)
    if dist.get_rank() == 0:
        path.mkdir(parents=True, exist_ok=True)
    dist.barrier()
    part, part_bytes, parts, skipped, total = {}, 0, 0, [], 0
    stem = _file_stem()

    def flush():
        nonlocal part, part_bytes, parts
        if writer and part:
            save_file(part, str(path / f"{stem}-part{parts:03d}.safetensors"), metadata={"format": FORMAT})
            parts += 1
        part, part_bytes = {}, 0

    device = torch.device("cuda", torch.cuda.current_device())  # NCCL; some params may live on the host
    host = [key for key, param in _named_params(model) if param.device.type != "cuda"]
    if host and dist.get_rank() == 0:
        logger.info("main-param save: %d params not on GPU, e.g. %s", len(host), host[:8])
    for key, param in _named_params(model):
        shard = owned.get(param)
        covered = torch.tensor([shard[1].size if shard else 0], dtype=torch.int64, device=device)
        dist.all_reduce(covered, group=group)
        if covered.item() == 0:
            skipped.append(key)  # not optimised (frozen): the HF weights are authoritative
            continue
        if covered.item() != param.numel():
            raise RuntimeError(f"{key}: optimizer shards cover {covered.item()} of {param.numel()} elements")
        full = torch.zeros(param.numel(), dtype=torch.float32, device=device)
        if shard:
            main, rng = shard
            full[rng.start : rng.end].copy_(main.detach().view(-1).float().to(device))
        dist.reduce(full, dst=dst, group=group)
        if writer:
            part[key] = full.view(param.shape).cpu()
            part_bytes += full.numel() * 4
            total += full.numel()
            if part_bytes >= _PART_BYTES:
                flush()
        del full
    flush()
    parallel = get_parallel_state()
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, {"stem": stem, "writer": writer, "parts": parts, "elements": total})
    if dist.get_rank() == 0:
        files = {g["stem"]: g["parts"] for g in gathered if g["writer"]}
        layout = {
            "format": FORMAT,
            "tp": parallel.tp.size,
            "pp": parallel.pp.size,
            "elements": sum(g["elements"] for g in gathered if g["writer"]),
            "skipped": skipped,
            "files": files,
        }
        (path / "layout.json").write_text(json.dumps(layout, indent=2) + "\n")
        logger.info("saved fp32 main params to %s: %s", path, {k: v for k, v in layout.items() if k != "skipped"})
    dist.barrier()
    return {"path": str(path), "skipped": len(skipped)}


def load_main_params(model, optimizer, path: str, *, max_mismatch_fraction: float = 1e-4) -> dict:
    """Overwrite this rank's fp32 main-param shards from ``path`` (written by ``save_main_params``).

    Checks every loaded tensor slice: more than ``max_mismatch_fraction`` of its elements with bf16 model
    weights != bf16(main) means the HF weights do not come from these main params (raise). Saved keys
    that no model parameter consumes are reported (and raise): the layouts differ."""
    from safetensors import safe_open

    path = Path(path)
    layout = json.loads((path / "layout.json").read_text())
    parallel = get_parallel_state()
    tp, pp = parallel.tp.size, parallel.pp.size
    if layout.get("format") != FORMAT or (layout["tp"], layout["pp"]) != (tp, pp):
        raise ValueError(f"{path}: saved with tp={layout.get('tp')} pp={layout.get('pp')}, running tp={tp} pp={pp}")
    stem = _file_stem()
    handles = [safe_open(str(f), framework="pt", device="cpu") for f in sorted(path.glob(f"{stem}-part*.safetensors"))]
    if not handles:
        raise FileNotFoundError(f"{path}: no {stem}-part*.safetensors")
    index = {key: h for h in handles for key in h.keys()}
    owned, _ = _owned(optimizer)
    model_keys = set()
    loaded = mismatched = elements = 0
    max_diff = 0.0
    bad_tensors = []
    with torch.no_grad():
        for key, param in _named_params(model):
            model_keys.add(key)
            shard = owned.get(param)
            if shard is None:
                continue
            if key not in index:
                raise KeyError(f"{path}: no saved main param for {key}")
            main, rng = shard
            if rng.size == 0:
                continue
            view = index[key].get_slice(key)
            shape = view.get_shape()
            if list(shape) != list(param.shape):
                raise ValueError(f"{key}: saved shape {shape} != param shape {list(param.shape)}")
            row = math.prod(shape[1:]) if len(shape) > 1 else 1
            first, last = rng.start // row, -(-rng.end // row)
            flat = view[first:last].reshape(-1)
            offset = rng.start - first * row
            values = flat[offset : offset + rng.size].to(device=main.device, dtype=torch.float32)
            main.view(-1).copy_(values)
            model_values = param.detach().view(-1)[rng.start : rng.end]
            if model_values.dtype != torch.float32:
                diff = model_values.float() - values.to(device=model_values.device, dtype=model_values.dtype).float()
                bad = int((diff != 0).sum().item())
                mismatched += bad
                if bad:
                    max_diff = max(max_diff, float(diff.abs().max().item()))
                    if bad > max_mismatch_fraction * rng.size:
                        bad_tensors.append((key, bad, rng.size))
            loaded += 1
            elements += rng.size
    unconsumed = sorted(set(index) - model_keys)
    stats = torch.tensor(
        [loaded, elements, mismatched, len(bad_tensors), len(unconsumed)],
        dtype=torch.float64,
        device=torch.cuda.current_device(),
    )
    dist.all_reduce(stats)
    worst = torch.tensor([max_diff], dtype=torch.float64, device=stats.device)
    dist.all_reduce(worst, op=dist.ReduceOp.MAX)
    summary = {
        "path": str(path),
        "shards_loaded": int(stats[0].item()),
        "elements": int(stats[1].item()),
        "bf16_mismatched_elements": int(stats[2].item()),
        "bf16_max_abs_diff": float(worst.item()),
        "tensors_over_tolerance": int(stats[3].item()),
        "unconsumed_saved_keys": int(stats[4].item()),
    }
    if bad_tensors:
        logger.error("main params vs bf16 weights over tolerance on this rank: %s", bad_tensors[:8])
    if unconsumed:
        logger.error("saved main params no model parameter consumes on this rank: %s", unconsumed[:8])
    if dist.get_rank() == 0:
        logger.info("loaded fp32 main params: %s", summary)
    if summary["tensors_over_tolerance"] or summary["unconsumed_saved_keys"]:
        raise RuntimeError(f"fp32 main params do not belong to these weights / this layout: {summary}")
    return summary


def optimizer_state_summary(model, optimizer) -> dict:
    """Collective health check after loading a Megatron checkpoint with optimizer state: bf16 model
    weights must equal bf16(main params), and Adam moments must be non-zero (i.e. restored, not fresh)."""
    owned, _ = _owned(optimizer)
    device = torch.device("cuda", torch.cuda.current_device())
    mismatched = elements = 0
    moment = torch.zeros(2, dtype=torch.float64, device=device)
    steps = set()
    with torch.no_grad():
        for opt in _optimizers(optimizer):
            inner = getattr(opt, "optimizer", None)
            for group in getattr(inner, "param_groups", []):
                if "step" in group:
                    steps.add(float(group["step"]))
            state = getattr(inner, "state", {})
            for shard_group in opt.shard_fp32_from_float16_groups + opt.shard_fp32_groups:
                for shard in shard_group:
                    st = state.get(shard, {})
                    for i, name in enumerate(("exp_avg", "exp_avg_sq")):
                        if name in st:
                            moment[i] += st[name].float().abs().sum().to(device)
                    if "step" in st:
                        steps.add(float(st["step"]))
        for _, param in _named_params(model):
            shard = owned.get(param)
            if shard is None or shard[1].size == 0 or param.dtype == torch.float32:
                continue
            main, rng = shard
            model_values = param.detach().view(-1)[rng.start : rng.end]
            mismatched += int(
                (model_values != main.detach().view(-1).to(model_values.device, model_values.dtype)).sum()
            )
            elements += rng.size
    counts = torch.tensor([mismatched, elements], dtype=torch.float64, device=device)
    dist.all_reduce(counts)
    dist.all_reduce(moment)
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, sorted(steps))
    summary = {
        "bf16_mismatched_elements": int(counts[0].item()),
        "elements": int(counts[1].item()),
        "sum_abs_exp_avg": float(moment[0].item()),
        "sum_abs_exp_avg_sq": float(moment[1].item()),
        "adam_steps": sorted({s for g in gathered for s in g}),
    }
    if dist.get_rank() == 0:
        logger.info("optimizer state: %s", summary)
    return summary
