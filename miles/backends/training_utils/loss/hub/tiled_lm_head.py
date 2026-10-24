"""Per-token log-probs and entropy from the LM-head input, never holding a microbatch of logits.

Under ``--log-probs-backend fused_lm_head`` the model returns its final hidden states (``lm_head.LMHeadInputs``)
instead of logits. The scored rows are projected ``--log-probs-chunk-size`` rows at a time and each
``[rows, V / TP]`` tile goes through the fused log-prob op (``fused_log_probs``). Under gradients a
tile's projection and softmax statistics are checkpointed together, so across tiles only the scored
hidden rows and a few fp32 numbers per row persist; the backward recomputes one tile at a time. Without
gradients the same tiles run unchecked, so no-grad scores equal the training forward's.

``tiled_cross_entropy`` scores every row of a sequence the same way, for Megatron's MTP loss.
"""

import functools
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from miles.backends.training_utils.loss.hub.fused_log_probs import fused_log_probs_and_entropy
from miles.backends.training_utils.loss.hub.lm_head import LMHeadProjection


@dataclass(frozen=True)
class _TileOptions:
    """The non-tensor inputs of one tile; immutable, so a checkpoint's replay sees the forward's values."""

    projection: LMHeadProjection
    vocab_size: int | None
    temperature: float
    with_entropy: bool
    entropy_requires_grad: bool


def tiled_log_probs_and_entropy(
    hidden_states: Tensor,
    rows: Tensor,
    targets: Tensor,
    *,
    projection: LMHeadProjection,
    tile_rows: int,
    vocab_size: int | None,
    temperature: float,
    with_entropy: bool,
    entropy_requires_grad: bool,
) -> tuple[Tensor, Tensor | None]:
    """Fp32 ``[R]`` log-probs of ``targets`` (and entropies) at ``rows`` of the ``[B * S]`` logits rows.

    ``hidden_states`` is ``[S, B, H]`` from ``projection.gather_sequence``; ``rows`` index the logits as
    the torch backend flattens them, ``[B, S]``. Unscored rows are never projected.
    """
    if tile_rows <= 0:
        raise ValueError(f"the LM-head tile needs a positive number of rows, got {tile_rows}")
    if rows.numel() == 0:
        # nothing to project; the losses attach such a rank to the head with ``lm_head.zero_loss_anchor``
        empty = hidden_states.new_empty(0, dtype=torch.float32)
        return empty, (empty if with_entropy else None)
    scored = _scored_hidden_rows(hidden_states, rows)
    options = _TileOptions(
        projection=projection,
        vocab_size=vocab_size,
        temperature=temperature,
        with_entropy=with_entropy,
        entropy_requires_grad=entropy_requires_grad,
    )
    score_tile = functools.partial(_score_tile, options=options)
    if torch.is_grad_enabled():
        scored = projection.track_weight_grad(scored)
        # each call holds its own tile and options, so the backward replays exactly the forward's tile
        score_tile = functools.partial(checkpoint, score_tile, use_reentrant=False, preserve_rng_state=False)
    tiles = [score_tile(hidden_tile, target_tile) for hidden_tile, target_tile in _tiles(scored, targets, tile_rows)]
    log_probs = torch.cat([log_prob for log_prob, _ in tiles])
    entropy = torch.cat([entropy for _, entropy in tiles]) if with_entropy else None
    return log_probs, entropy


def tiled_cross_entropy(
    hidden_states: Tensor,
    labels: Tensor,
    *,
    projection: LMHeadProjection,
    tile_rows: int,
    ignore_index: int,
) -> Tensor:
    """Fp32 ``[B, S]`` cross-entropy of ``labels`` ``[B, S]`` from the ``[S, B, H]`` hidden states, tile by tile.

    Megatron's linear cross-entropy contract, as its MTP loss calls it: every row is projected, the softmax
    spans the whole padded vocabulary as Megatron's cross-entropy does, and an ``ignore_index`` label
    costs nothing. ``hidden_states`` are sequence-sharded under SP, as the model produces them.
    """
    hidden_states = projection.gather_sequence(hidden_states)
    seq_len, batch_size, _ = hidden_states.shape
    assert labels.shape == (batch_size, seq_len), f"{tuple(labels.shape)} labels for [{seq_len}, {batch_size}] rows"
    labels = labels.reshape(-1)  # the logits rows' order, [B * S]
    ignored = labels == ignore_index
    log_probs, _ = tiled_log_probs_and_entropy(
        hidden_states,
        torch.arange(labels.numel(), device=labels.device),
        labels.masked_fill(ignored, 0),
        projection=projection,
        tile_rows=tile_rows,
        vocab_size=None,
        temperature=1.0,
        with_entropy=False,
        entropy_requires_grad=False,
    )
    return (-log_probs).masked_fill(ignored, 0.0).view(batch_size, seq_len)


def _scored_hidden_rows(hidden_states: Tensor, rows: Tensor) -> Tensor:
    """``[R, H]``: the hidden rows that score, in ``rows`` order; one copy, made before tiling."""
    seq_len, batch_size, width = hidden_states.shape
    # logits row b * S + s is hidden row s * B + b: Megatron keeps hidden states sequence-major
    hidden_rows = rows if batch_size == 1 else (rows % seq_len) * batch_size + rows // seq_len
    return hidden_states.reshape(seq_len * batch_size, width).index_select(0, hidden_rows)


def _tiles(scored: Tensor, targets: Tensor, tile_rows: int) -> list[tuple[Tensor, Tensor]]:
    # one split node per tensor: its backward concatenates the tile gradients instead of adding one
    # zero-padded [R, H] gradient per tile
    return list(zip(scored.split(tile_rows), targets.split(tile_rows), strict=True))


def _score_tile(hidden_tile: Tensor, targets: Tensor, *, options: _TileOptions) -> tuple[Tensor, Tensor | None]:
    logits = options.projection.project(hidden_tile)
    rows = torch.arange(logits.size(0), device=logits.device)
    return fused_log_probs_and_entropy(
        logits,
        rows,
        targets,
        tp_group=options.projection.tp_group,
        vocab_size=options.vocab_size,
        temperature=options.temperature,
        with_entropy=options.with_entropy,
        entropy_requires_grad=options.entropy_requires_grad,
        # nothing else reads a tile's logits: under a checkpoint the backward writes into its replayed copy
        inplace_backward=True,
    )
