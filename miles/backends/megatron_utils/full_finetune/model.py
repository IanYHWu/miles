"""Full fine-tuning operations over the Megatron model and its optimizer.

Gradients accumulate in the DDP buffers across work units until an optimizer step. Megatron reduces them across
ranks in ``finalize_model_grads``, whose reduce-scatter under the distributed optimizer and sums of
sequence-parallel norm gradients do not tolerate running twice over one window, so a backward pass only marks the
reduction pending and the optimizer step runs it once.
"""

import math
from argparse import Namespace
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from megatron.core.distributed import DistributedDataParallel as DDP
from megatron.core.optimizer.optimizer import ChainedOptimizer

from miles.backends.megatron_utils.model import finalize_model_grads_with_empty_cache, save
from miles.backends.training_utils.checkpoint.io import write_checkpoint_dir
from miles.backends.training_utils.model_companion import ModelCompanionWeightVersionUtils
from miles.utils.argparse_utils import inplace_modify_args

# each checkpoint directory holds one Megatron iteration
_CHECKPOINT_ITERATION = 1


@dataclass
class GradientWindow:
    """The gradients work units have accumulated since the last optimizer step."""

    reduction_pending: bool = False

    def defer_reduction(self, *_args, **_kwargs) -> None:
        """Megatron's ``finalize_model_grads_func`` for a work unit: the reduction waits for the optimizer step."""
        self.reduction_pending = True


def optim_step(
    model: Sequence[DDP], optimizer: ChainedOptimizer, window: GradientWindow, adam_params: dict
) -> dict[str, float]:
    """Step on the window's gradients under this step's AdamParams, then drop them.

    -> ``{"grad_norm": x}`` stepped, ``{"skipped_nonfinite": 1.0}`` dropped, as the multi-LoRA slots report.
    bf16 has no loss scaler to catch a non-finite gradient, and clipping by a non-finite norm would write it into
    the weights, so the norm is checked before the step.
    """
    if window.reduction_pending:
        finalize_model_grads_with_empty_cache(model, None)
        window.reduction_pending = False
    _apply_adam_params(optimizer, adam_params)
    found_inf = optimizer.prepare_grads()
    grad_norm = float(optimizer.get_grad_norm())
    if found_inf or not math.isfinite(grad_norm):
        outcome = {"skipped_nonfinite": 1.0}
    else:
        update_successful, grad_norm, _ = optimizer.step()
        assert update_successful, "the optimizer refused a step on finite gradients"
        ModelCompanionWeightVersionUtils.bump_weight_version(model)
        outcome = {"grad_norm": float(grad_norm)}
    discard_gradients(model, optimizer, window)
    return outcome


def _apply_adam_params(optimizer: ChainedOptimizer, adam_params: dict) -> None:
    """Write AdamParams into every param group, scaling weight decay by the group's ``wd_mult`` as Megatron's
    scheduler does (0 for norms and biases); startup refuses per-group learning rates.

    A ``grad_clip_norm`` of 0 clips nothing, in Tinker and in Megatron.
    """
    for group in optimizer.param_groups:
        if torch.is_tensor(group["lr"]):
            group["lr"].fill_(adam_params["learning_rate"])
        else:
            group["lr"] = adam_params["learning_rate"]
        group["betas"] = (adam_params["beta1"], adam_params["beta2"])
        group["eps"] = adam_params["eps"]
        group["weight_decay"] = adam_params["weight_decay"] * group.get("wd_mult", 1.0)
    for child in optimizer.chained_optimizers:
        child.config.clip_grad = adam_params["grad_clip_norm"]


def discard_gradients(model: Sequence[DDP], optimizer: ChainedOptimizer, window: GradientWindow) -> None:
    """Drop the window: the DDP buffers, the optimizer's copies of them and any pending reduction."""
    for model_chunk in model:
        model_chunk.zero_grad_buffer()
    optimizer.zero_grad()
    window.reduction_pending = False


def save_checkpoint(
    args: Namespace, model: Sequence[DDP], optimizer: ChainedOptimizer, path: str, metadata: dict | None
) -> None:
    """Weights and optimizer state as a Megatron checkpoint under ``path``, beside the gateway's ``META.json``.

    Loading it under another parallel layout is untested.
    """

    def write_shards(checkpoint_dir: Path) -> None:
        with inplace_modify_args(args, {"save": str(checkpoint_dir), "async_save": False}):
            save(_CHECKPOINT_ITERATION, model, optimizer, None)

    write_checkpoint_dir(path, write_shards, metadata=metadata)
