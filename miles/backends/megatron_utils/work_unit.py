"""One Tinker work unit through the Megatron model, shared by the multi-LoRA and full fine-tuning actors.

A work unit is a single forward pass, and a backward pass whose gradients add to the ones already accumulated;
the optimizer step that consumes them is a separate command.
"""

from argparse import Namespace
from collections.abc import Callable, Sequence

from megatron.core.distributed import DistributedDataParallel as DDP

from miles.backends.megatron_utils.model import run_forward_backward_pass, setup_train_iteration_config
from miles.backends.training_utils.data.rollout import get_data_iterator
from miles.backends.training_utils.metrics.log_utils import aggregate_train_losses
from miles.backends.training_utils.parallel import get_parallel_state
from miles.utils.dumper_utils import DumperMegatronUtil, DumperPhase
from miles.utils.types import RolloutBatch


def run_forward_backward(
    args: Namespace,
    batch_id: int,
    model: Sequence[DDP],
    rollout_data: RolloutBatch,
    *,
    forward_only: bool = False,
    finalize_model_grads_func: Callable | None = None,
) -> dict:
    """``finalize_model_grads_func`` replaces the gradient reductions Megatron runs after the backward pass."""
    data_iterator, num_microbatches = get_data_iterator(args, model, rollout_data)
    assert len(num_microbatches) == 1, "a work unit is a single forward/backward pass"

    for iterator in data_iterator:
        iterator.reset()
    for model_chunk in model:
        model_chunk.train()
    # disable_optimizer: bf16 loss scaling is the identity, so the pass needs no optimizer
    config = setup_train_iteration_config(args, model, None, disable_optimizer=True)
    if finalize_model_grads_func is not None:
        config.finalize_model_grads_func = finalize_model_grads_func
    if not forward_only:
        reset_grad_metadata_keep_grads(model)

    dumper_phase_util = DumperMegatronUtil(args, model, DumperPhase.FWD_BWD, rollout_id=batch_id)
    losses_reduced = run_forward_backward_pass(
        args,
        dumper_phase_util,
        data_iterator,
        model,
        num_microbatches[0],
        num_rollouts=None,
        forward_only=forward_only,
    )
    per_datum_outputs = [output for microbatch in losses_reduced for output in microbatch["per_datum"]]
    dumper_phase_util.finalize(model)

    if get_parallel_state().is_pp_last_stage:
        return {"metrics": aggregate_train_losses(losses_reduced, None), "per_datum": per_datum_outputs}
    return {"metrics": {}, "per_datum": per_datum_outputs}


def reset_grad_metadata_keep_grads(model_chunks) -> None:
    """Reset DDP bookkeeping while retaining the accumulated gradients."""
    for model_chunk in model_chunks:
        if getattr(model_chunk.config, "cuda_graph_impl", "none") != "transformer_engine":
            for param in model_chunk.params_with_grad:
                param.grad_added_to_main_grad = False
        for bucket_group in model_chunk.bucket_groups + model_chunk.expert_parallel_bucket_groups:
            bucket_group.reset()
