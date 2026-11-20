"""Multi-LoRA slot operations; the work-unit pass itself is ``megatron_utils.work_unit``."""

from argparse import Namespace
from collections.abc import Sequence

from megatron.core.distributed import DistributedDataParallel as DDP

from miles.backends.megatron_utils.lora.optimizer import SlotOptimizer, step_slot_optimizers


def optim_step(
    slot_optimizers: dict[int, SlotOptimizer],
    adam_params_by_slot: dict[int, dict],
) -> dict[int, dict]:
    stepped = {slot: slot_optimizers[slot] for slot in adam_params_by_slot}
    return step_slot_optimizers(stepped, adam_params_by_slot)


def load_slot(args: Namespace, model: Sequence[DDP], slot: int, rank: int, alpha: float) -> SlotOptimizer:
    from megatron.bridge.peft.multi_lora_layers import init_adapter_slot

    init_adapter_slot(model, slot, rank=rank, alpha=alpha)
    slot_optimizer = SlotOptimizer(args, model, slot)
    slot_optimizer.zero_grads()
    return slot_optimizer


def unload_slot(model: Sequence[DDP], slot_optimizer: SlotOptimizer) -> None:
    """The caller drops the instance afterwards; its state dies with it."""
    from megatron.bridge.peft.multi_lora_layers import clear_adapter_slot

    clear_adapter_slot(model, slot_optimizer.slot)
    slot_optimizer.zero_grads()
