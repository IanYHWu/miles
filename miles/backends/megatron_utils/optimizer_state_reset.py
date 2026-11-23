"""Optimizer history schema and reset, failing closed on state it does not know.

Used by --reset-optimizer-states and by optimizer_state_swap, which moves the same history.

The Muon classes are imported where they are matched: Megatron builds without
``emerging_optimizers`` (the NPU pin) still run Adam.
"""

from collections.abc import Iterator

import torch
from megatron.core.optimizer import Adam, CPUAdam


class UnsupportedOptimizerState(RuntimeError):
    """An optimizer or state key this module has no schema for; teach history_keys rather than guessing."""


def reset_optimizer_states(optimizer) -> None:
    for leaf in leaf_optimizers(optimizer):
        _reset(leaf, history_keys(leaf))


def leaf_optimizers(optimizer) -> Iterator[torch.optim.Optimizer]:
    """Descend Megatron's wrappers to the torch optimizers that own the state."""
    if optimizer is None:
        return
    for children in ("chained_optimizers", "sub_optimizers"):
        if hasattr(optimizer, children):
            for child in getattr(optimizer, children):
                yield from leaf_optimizers(child)
            return
    if hasattr(optimizer, "optimizer"):
        yield from leaf_optimizers(optimizer.optimizer)
    else:
        yield optimizer


def history_keys(optimizer) -> frozenset[str]:
    """The per-parameter state keys that hold an optimizer's history (moments, momentum buffers)."""
    if isinstance(optimizer, (Adam, CPUAdam)):
        return frozenset({"exp_avg", "exp_avg_sq"})
    from megatron.core.optimizer.emerging_optimizers import TensorParallelAdaptiveMuon, TensorParallelMuon

    if isinstance(optimizer, TensorParallelAdaptiveMuon):
        return frozenset({"momentum_buffer", "moment2_buffer"})
    if isinstance(optimizer, TensorParallelMuon):
        return frozenset({"momentum_buffer"})
    raise UnsupportedOptimizerState(f"no reset schema for {type(optimizer).__name__}")


def _reset(optimizer, history: frozenset[str]) -> None:
    for group in optimizer.param_groups:
        _reset_step(group)
    for state in optimizer.state.values():
        check_state_keys(optimizer, state, history)
        _reset_step(state)
        for key in history & set(state):
            state[key].zero_()


def check_state_keys(optimizer, state: dict, history: frozenset[str]) -> None:
    unknown = set(state) - history - {"step", "master_param"}
    if unknown:
        raise UnsupportedOptimizerState(f"unknown state keys {sorted(unknown)} on {type(optimizer).__name__}")


def _reset_step(container) -> None:
    step = container.get("step")
    if isinstance(step, torch.Tensor):
        step.zero_()
    elif isinstance(step, (int, float)):
        container["step"] = 0
    elif step is not None:
        raise UnsupportedOptimizerState(f"unsupported step type {type(step).__name__}")
