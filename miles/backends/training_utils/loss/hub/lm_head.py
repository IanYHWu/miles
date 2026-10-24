"""The model output under ``--log-probs-backend fused_lm_head``: the LM head's input instead of its logits.

``tiled_lm_head`` turns it into per-token log-probs; this module stays free of the Triton op so the
losses can name the type on hosts without Triton.
"""

from dataclasses import dataclass
from typing import Literal, Protocol

import torch
import torch.distributed as dist
from torch import Tensor

LMHeadLogitsDtype = Literal["model", "fp32"]


class LMHeadProjection(Protocol):
    """The vocab-parallel output layer of one forward call, applied to rows of its hidden states."""

    tp_group: dist.ProcessGroup | None
    logits_dtype: LMHeadLogitsDtype

    def gather_sequence(self, hidden_states: Tensor) -> Tensor:
        """``[S, B, H]`` hidden states of the whole local sequence; the backward sums their gradient over TP."""
        ...

    def project(self, hidden_rows: Tensor) -> Tensor:
        """``[rows, V / TP]`` logits of ``[rows, H]`` hidden rows, in ``logits_dtype``."""
        ...

    def track_weight_grad(self, hidden_rows: Tensor) -> Tensor:
        """``hidden_rows`` unchanged, for checkpointed tiles to project; the weight's gradient bookkeeping
        completes in the backward after every tile's."""
        ...


@dataclass(frozen=True)
class LMHeadInputs:
    """The policy model's output under ``fused_lm_head``: final hidden states and the projection to logits."""

    hidden_states: Tensor  # [S, B, H] as the decoder returns them, sequence-sharded under SP
    projection: LMHeadProjection

    @property
    def device(self) -> torch.device:
        return self.hidden_states.device


def zero_loss_anchor(model_output: Tensor | LMHeadInputs) -> Tensor:
    """``0`` attached to the model output, so a rank that scores nothing still reaches every parameter.

    Logits reach the output layer and the backbone; hidden states reach them through one projected row.
    In fp32: an fp16 sum can overflow to inf, and ``0 * inf`` is nan.
    """
    if isinstance(model_output, LMHeadInputs):
        projection = model_output.projection
        hidden_states = projection.gather_sequence(model_output.hidden_states)
        model_output = projection.project(hidden_states.flatten(0, 1)[:1])
    return 0 * model_output.sum(dtype=torch.float32)
