"""CP token-layout helpers used inside model plugin ``forward`` implementations.

Distinct from ``miles.backends.training_utils.data.context_parallel``, which owns the CP
helpers the training backend applies *around* the model (slicing data, masks,
logprobs and logits). Everything here runs inside a layer.

Megatron CP stores each rank's tokens in the zigzag load-balanced order that
ring attention wants, while fla's CP operators expect a contiguous rank-local
chunk. Anything that hands a sequence to fla under CP therefore has to relayout
first and undo it afterwards.
"""

import os

import torch
import torch.distributed as dist
import torch.nn as nn

try:
    from fla.ops.cp import build_cp_context as _fla_build_cp_context
except ImportError:
    _fla_build_cp_context = None


def build_fla_cp_context(cu_seqlens: torch.Tensor, cp_group, conv_kernel_size: int, device: torch.device):
    """fla CP context for a rank of ``cp_group`` from the global packed boundaries ``cu_seqlens``."""
    if _fla_build_cp_context is None:
        raise RuntimeError(
            "Hybrid CP requires fla.ops.cp (flash-linear-attention >= 0.4.2) " "but it could not be imported."
        )
    if cu_seqlens is None or cu_seqlens.numel() < 2:
        raise ValueError(f"Hybrid CP requires valid cu_seqlens (at least 2 elements) but got {cu_seqlens}")
    return _fla_build_cp_context(
        cu_seqlens=cu_seqlens.to(device=device, dtype=torch.int32),
        group=cp_group,
        conv1d_kernel_size=conv_kernel_size,
    )


def build_gdn_cp_context(module: nn.Module, cu_seqlens: torch.Tensor, device: torch.device):
    """Build fla CP context for a GatedDeltaNet module from packed sequence boundaries.

    Args:
        module: GDN module with ``cp_group`` / ``cp_world_size`` / ``conv_kernel_size``.
        cu_seqlens: Global packed sequence boundaries (e.g. ``packed_seq_params.cu_seqlens_q``).
        device: Target device.

    Returns ``None`` when CP is not configured on the module (``cp_group`` not set).
    """
    cp_group = getattr(module, "cp_group", None)
    if cp_group is None:
        return None
    return build_fla_cp_context(cu_seqlens, cp_group, module.conv_kernel_size, device)


def _zigzag_positions(cu_seqlens, cp_size, total):
    """For every position ``p`` of a ``total``-token packed stream, its row in the rank-concatenated zigzag
    shards, and the shard size.

    Rank ``r``'s zigzag shard holds, per sequence, chunks ``r`` and ``2 * cp_size - 1 - r`` of
    ``2 * cp_size`` equal chunks, or its contiguous ``1 / cp_size`` when the length is not a multiple of
    ``2 * cp_size`` (the final padding)."""
    cu = cu_seqlens.to(torch.int64)
    lengths = cu[1:] - cu[:-1]
    torch._assert_async(cu[-1] == total)
    torch._assert_async((lengths % cp_size == 0).all())
    shard = total // cp_size
    position = torch.arange(total, device=cu.device)
    seq = torch.searchsorted(cu, position, right=True) - 1
    start = cu[seq]
    length = lengths[seq]
    offset = position - start
    zigzag = length % (2 * cp_size) == 0
    chunk = torch.where(zigzag, length // (2 * cp_size), length // cp_size)
    part = offset // chunk
    mirrored = zigzag & (part >= cp_size)
    owner = torch.where(mirrored, 2 * cp_size - 1 - part, part)
    return owner * shard + start // cp_size + offset % chunk + mirrored * chunk, shard


def _relayout_indices(cu_seqlens, cp_rank, cp_size, total):
    """Index maps between the two layouts of a ``total``-token packed stream, built on device.

    Returns ``(to_packed, to_zigzag)``: ``to_packed`` selects this rank's contiguous shard from the
    rank-concatenated zigzag shards, ``to_zigzag`` selects this rank's zigzag shard from the
    rank-concatenated contiguous shards (the packed stream itself)."""
    zigzag_position, shard = _zigzag_positions(cu_seqlens, cp_size, total)
    packed_position = torch.empty_like(zigzag_position)
    packed_position[zigzag_position] = torch.arange(total, device=zigzag_position.device)
    rows = slice(cp_rank * shard, (cp_rank + 1) * shard)
    return zigzag_position[rows], packed_position[rows]


def _a2a_plan(cu_seqlens, cp_rank, cp_size, total):
    """All-to-all plan between this rank's zigzag shard and its contiguous (packed) shard.

    The two layouts are a permutation, so instead of all-gathering the whole stream to every rank each
    rank sends only the rows another rank needs. Returns ``(send_index, send_counts, recv_order,
    recv_counts)``: zigzag -> packed sends ``x[send_index]`` split by ``send_counts`` (to ranks
    0..cp_size-1), and the rows received, split by ``recv_counts`` (from ranks 0..cp_size-1), go to packed
    rows ``recv_order``. Counts are host lists."""
    zigzag_position, shard = _zigzag_positions(cu_seqlens, cp_size, total)
    src_rank = zigzag_position // shard  # zigzag owner of packed position p
    dst_rank = torch.arange(total, device=zigzag_position.device) // shard  # packed owner of position p
    # sender side: my zigzag rows in packed order (dst_rank is non-decreasing in p, so grouped by dst)
    mine = src_rank == cp_rank
    send_index = zigzag_position[mine] - cp_rank * shard
    send_counts = torch.bincount(dst_rank[mine], minlength=cp_size)
    # receiver side: my packed rows grouped by source rank (stable keeps packed order within a source)
    sources = src_rank[cp_rank * shard : (cp_rank + 1) * shard]
    recv_order = torch.sort(sources, stable=True).indices
    recv_counts = torch.bincount(sources, minlength=cp_size)
    counts = torch.stack([send_counts, recv_counts]).cpu().tolist()
    return send_index, counts[0], recv_order, counts[1]


_A2A_PLAN_CACHE_SIZE = 8
_a2a_plan_cache: dict = {}


def _cached_a2a_plan(cu_seqlens, cp_rank, cp_size, total):
    """Every layer of a microbatch relayouts the same stream: build its plan once."""
    key = (tuple(cu_seqlens.tolist()), cp_rank, cp_size, total, cu_seqlens.device)
    plan = _a2a_plan_cache.get(key)
    if plan is None:
        if len(_a2a_plan_cache) >= _A2A_PLAN_CACHE_SIZE:
            _a2a_plan_cache.clear()
        plan = _a2a_plan_cache[key] = _a2a_plan(cu_seqlens, cp_rank, cp_size, total)
    return plan


def _all_to_all_rows(x, send_rows, send_counts, recv_counts, group):
    out = x.new_empty((sum(recv_counts), *x.shape[1:]))
    dist.all_to_all_single(
        out,
        x.index_select(0, send_rows).contiguous(),
        output_split_sizes=recv_counts,
        input_split_sizes=send_counts,
        group=group,
    )
    return out


def _a2a_relayout(x, plan, cp_group, to_packed):
    send_index, send_counts, recv_order, recv_counts = plan
    if to_packed:
        received = _all_to_all_rows(x, send_index, send_counts, recv_counts, cp_group)
        out = torch.empty_like(received)
        out.index_copy_(0, recv_order, received)
        return out
    # packed -> zigzag: the inverse permutation
    received = _all_to_all_rows(x, recv_order, recv_counts, send_counts, cp_group)
    out = torch.empty_like(received)
    out.index_copy_(0, send_index, received)
    return out


class _A2ARelayout(torch.autograd.Function):
    """zigzag -> packed (``to_packed``) or packed -> zigzag by a sparse all-to-all; the gradient is the
    opposite permutation."""

    @staticmethod
    def forward(ctx, x, plan, cp_group, to_packed):
        ctx.plan, ctx.cp_group, ctx.to_packed = plan, cp_group, to_packed
        return _a2a_relayout(x, plan, cp_group, to_packed)

    @staticmethod
    def backward(ctx, grad_output):
        return _a2a_relayout(grad_output.contiguous(), ctx.plan, ctx.cp_group, not ctx.to_packed), None, None, None


def _gather_select(x, index, cp_group):
    gathered = x.new_empty((x.shape[0] * dist.get_world_size(group=cp_group), *x.shape[1:]))
    dist.all_gather_into_tensor(gathered, x.contiguous(), group=cp_group)
    return gathered.index_select(0, index)


class _Relayout(torch.autograd.Function):
    """All-gather over CP, then select this rank's rows; the gradient is the same with the inverse map."""

    @staticmethod
    def forward(ctx, x, index, inverse_index, cp_group):
        ctx.cp_group = cp_group
        ctx.save_for_backward(inverse_index)
        return _gather_select(x, index, cp_group)

    @staticmethod
    def backward(ctx, grad_output):
        (inverse_index,) = ctx.saved_tensors
        return _gather_select(grad_output, inverse_index, ctx.cp_group), None, None, None


def _use_all_to_all() -> bool:
    """The sparse all-to-all moves each row once; ``MILES_CP_RELAYOUT=allgather`` selects the all-gather
    relayout, which moves the whole stream to every rank."""
    return os.environ.get("MILES_CP_RELAYOUT", "a2a") == "a2a"


def _relayout(hidden_states, cu_seqlens, cp_group, cp_rank, cp_size, *, to_packed):
    total = hidden_states.size(0) * cp_size
    if _use_all_to_all():
        plan = _cached_a2a_plan(cu_seqlens, cp_rank, cp_size, total)
        return _A2ARelayout.apply(hidden_states, plan, cp_group, to_packed)
    to_packed_index, to_zigzag_index = _relayout_indices(cu_seqlens, cp_rank, cp_size, total)
    if to_packed:
        return _Relayout.apply(hidden_states, to_packed_index, to_zigzag_index, cp_group)
    return _Relayout.apply(hidden_states, to_zigzag_index, to_packed_index, cp_group)


def zigzag_to_packed_shard(hidden_states, cu_seqlens, cp_group, cp_rank, cp_size):
    return _relayout(hidden_states, cu_seqlens, cp_group, cp_rank, cp_size, to_packed=True)


def packed_shard_to_zigzag(hidden_states, cu_seqlens, cp_group, cp_rank, cp_size):
    return _relayout(hidden_states, cu_seqlens, cp_group, cp_rank, cp_size, to_packed=False)
