import gc
import logging

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


def clear_memory(clear_host_memory: bool = False):
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    if clear_host_memory:
        torch._C._host_emptyCache()


def available_memory():
    device = torch.cuda.current_device()
    free, total = torch.cuda.mem_get_info(device)
    return {
        "gpu": str(device),
        "total_GB": _byte_to_gb(total),
        "free_GB": _byte_to_gb(free),
        "used_GB": _byte_to_gb(total - free),
        "allocated_GB": _byte_to_gb(torch.cuda.memory_allocated(device)),
        "reserved_GB": _byte_to_gb(torch.cuda.memory_reserved(device)),
    }


def peak_memory(reset: bool = True) -> dict:
    """This rank's peak allocated / reserved GPU memory (GiB, unrounded) since the last reset."""
    device = torch.cuda.current_device()
    stats = {
        "rank": dist.get_rank(),
        "max_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
        "max_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
        "total_gib": torch.cuda.get_device_properties(device).total_memory / 2**30,
    }
    if reset:
        torch.cuda.reset_peak_memory_stats(device)
    return stats


def _byte_to_gb(n: int):
    return round(n / (1024**3), 2)


def print_memory(msg, clear_before_print: bool = False):
    if clear_before_print:
        clear_memory()

    memory_info = available_memory()
    # Need to print for all ranks, b/c different rank can have different behaviors
    logger.info(
        f"[Rank {dist.get_rank()}] Memory-Usage {msg}{' (cleared before print)' if clear_before_print else ''}: {memory_info}"
    )
    return memory_info
