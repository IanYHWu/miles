"""Several optimizer histories on one Megatron model, e.g. separate Adam states for two training phases.

Only one history lives in the optimizer (on the GPU); the others wait in host memory under a name.
``OptimizerStateStash.swap`` makes another one active: when both sides hold a history they are
exchanged tensor by tensor (no second host copy), otherwise the active one is copied out and the
optimizer starts fresh (zero history, step 0). A history is what ``optimizer_state_reset`` clears:
the per-parameter moments or momentum buffers and the step counters. The fp32 main params are
shared by every history and never touched here.
"""

import time
from pathlib import Path

import torch

from miles.backends.megatron_utils.optimizer_state_reset import (
    check_state_keys,
    history_keys,
    leaf_optimizers,
    reset_optimizer_states,
)

# A snapshot: {"leaves": [{"groups": [step per param group], "states": [{key: tensor, "step": step}]}],
# "gib": host bytes / 2**30, "seconds": copy time}, with leaves and states in optimizer order.
Snapshot = dict


def _history(leaf) -> frozenset[str]:
    history = history_keys(leaf)
    for state in leaf.state.values():
        check_state_keys(leaf, state, history)
    return history


def _params_with_state(leaf):
    for group in leaf.param_groups:
        for param in group["params"]:
            if param in leaf.state:
                yield param


def _host_copy(tensor: torch.Tensor) -> torch.Tensor:
    out = torch.empty(tensor.shape, dtype=tensor.dtype, device="cpu", pin_memory=torch.cuda.is_available())
    out.copy_(tensor, non_blocking=True)
    return out


def _copy_step(step):
    return step.detach().clone().cpu() if isinstance(step, torch.Tensor) else step


def _set_step(container: dict, step) -> None:
    current = container.get("step")
    if isinstance(current, torch.Tensor) and isinstance(step, torch.Tensor):
        current.copy_(step)
    elif step is not None:
        container["step"] = step


def _synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _snapshot(optimizer) -> Snapshot:
    start = time.time()
    leaves, nbytes = [], 0
    for leaf in leaf_optimizers(optimizer):
        history = _history(leaf)
        states = []
        for param in _params_with_state(leaf):
            state = leaf.state[param]
            entry = {key: _host_copy(state[key]) for key in history & set(state)}
            nbytes += sum(value.numel() * value.element_size() for value in entry.values())
            entry["step"] = _copy_step(state.get("step"))
            states.append(entry)
        leaves.append({"groups": [_copy_step(group.get("step")) for group in leaf.param_groups], "states": states})
    _synchronize()
    return {"leaves": leaves, "gib": nbytes / 2**30, "seconds": time.time() - start}


def _has_states(snapshot: Snapshot) -> bool:
    return any(leaf["states"] for leaf in snapshot["leaves"])


def _check_layout(optimizer, snapshot: Snapshot) -> None:
    """Raise before anything is written if the snapshot does not fit the optimizer's current state."""
    leaves = list(leaf_optimizers(optimizer))
    if len(leaves) != len(snapshot["leaves"]):
        raise RuntimeError(f"optimizer layout changed: {len(leaves)} leaves now, {len(snapshot['leaves'])} stashed")
    for leaf, saved in zip(leaves, snapshot["leaves"], strict=True):
        params = list(_params_with_state(leaf))
        if len(params) != len(saved["states"]) or len(leaf.param_groups) != len(saved["groups"]):
            raise RuntimeError(f"optimizer layout changed: {len(params)} states now, {len(saved['states'])} stashed")
        for param, entry in zip(params, saved["states"], strict=True):
            for key, value in entry.items():
                if key != "step" and tuple(leaf.state[param][key].shape) != tuple(value.shape):
                    raise RuntimeError(
                        f"optimizer state {key} shape {tuple(leaf.state[param][key].shape)} != "
                        f"stashed {tuple(value.shape)}"
                    )


def _state_owners(optimizer):
    """The optimizers below Megatron's chained/sub wrappers, each still wrapping its torch optimizer."""
    if optimizer is None:
        return
    for children in ("chained_optimizers", "sub_optimizers"):
        if hasattr(optimizer, children):
            for child in getattr(optimizer, children):
                yield from _state_owners(child)
            return
    yield optimizer


def _materialize(optimizer) -> bool:
    """Create the state tensors of an optimizer that has not stepped yet (TE FusedAdam allocates them
    lazily), so a stashed history can be written into them.

    Runs Megatron's dummy step with zero gradients at LR 0 and weight decay 0, so the weights cannot
    move, then zeroes the history. Returns whether anything was created."""
    leaves = list(leaf_optimizers(optimizer))
    if not leaves or all(leaf.state for leaf in leaves):
        return False
    for owner in _state_owners(optimizer):
        inner = getattr(owner, "optimizer", owner)
        if inner.state:
            continue
        saved = [(group.get("lr"), group.get("weight_decay")) for group in inner.param_groups]
        for group in inner.param_groups:
            if isinstance(group.get("lr"), torch.Tensor):
                group["lr"].fill_(0.0)
            else:
                group["lr"] = 0.0
            group["weight_decay"] = 0.0
        try:
            if hasattr(owner, "_init_optimizer_states_with_dummy_values"):
                owner._init_optimizer_states_with_dummy_values()
            else:  # a bare torch optimizer
                for group in inner.param_groups:
                    for param in group["params"]:
                        param.grad = torch.zeros_like(param)
                inner.step()
                inner.zero_grad()
        finally:
            for group, (lr, weight_decay) in zip(inner.param_groups, saved, strict=True):
                if isinstance(group.get("lr"), torch.Tensor):
                    group["lr"].copy_(lr) if isinstance(lr, torch.Tensor) else group["lr"].fill_(lr)
                else:
                    group["lr"] = lr
                group["weight_decay"] = weight_decay
    reset_optimizer_states(optimizer)
    return True


def _exchange(optimizer, snapshot: Snapshot) -> None:
    """Swap the active history with a stashed one in place; the snapshot then holds the previously
    active history. Peak extra memory is one tensor, not a second full snapshot."""
    _check_layout(optimizer, snapshot)
    for leaf, saved in zip(leaf_optimizers(optimizer), snapshot["leaves"], strict=True):
        previous_groups = []
        for group, step in zip(leaf.param_groups, saved["groups"], strict=True):
            previous_groups.append(_copy_step(group.get("step")))
            _set_step(group, step)
        saved["groups"] = previous_groups
        for param, entry in zip(_params_with_state(leaf), saved["states"], strict=True):
            state = leaf.state[param]
            for key in list(entry):
                if key == "step":
                    previous = _copy_step(state.get("step"))
                    _set_step(state, entry["step"])
                    entry["step"] = previous
                else:
                    active = state[key].clone()
                    state[key].copy_(entry[key], non_blocking=True)
                    entry[key].copy_(active, non_blocking=True)
                    del active
    _synchronize()


class OptimizerStateStash:
    """Named optimizer histories held in host memory, one rank's share of a distributed optimizer."""

    def __init__(self) -> None:
        self._snapshots: dict[str, Snapshot] = {}

    @property
    def names(self) -> list[str]:
        return sorted(self._snapshots)

    def swap(self, optimizer, *, stash_as: str, load: str) -> dict:
        """Stash the active history as ``stash_as`` and make ``load`` active (fresh if never stashed)."""
        start = time.time()
        target = self._snapshots.pop(load, None)
        if target is not None and _has_states(target):
            _materialize(optimizer)  # e.g. a phase whose optimizer never stepped since a restart
            _exchange(optimizer, target)  # target now holds the previously active history
            self._snapshots[stash_as] = target
            loaded, gib = "restored", target["gib"]
        else:
            snapshot = _snapshot(optimizer)
            self._snapshots[stash_as] = snapshot
            reset_optimizer_states(optimizer)
            loaded, gib = "fresh", snapshot["gib"]
        return {
            "stashed": stash_as,
            "stash_gib": round(gib, 2),
            "loaded": load,
            "load": loaded,
            "seconds": round(time.time() - start, 1),
        }

    def save(self, path: str | Path) -> dict:
        """Write the stashed histories to one file (atomically), e.g. beside a Megatron checkpoint,
        which holds only the active one."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(".tmp")
        torch.save(self._snapshots, partial)
        partial.replace(path)
        return {"states": self.names, "gib": round(sum(s["gib"] for s in self._snapshots.values()), 2)}

    def load(self, path: str | Path) -> None:
        """Replace the stashed histories with those ``save`` wrote."""
        # a file this class wrote next to the run's own checkpoint
        self._snapshots = torch.load(path, map_location="cpu", weights_only=False)
