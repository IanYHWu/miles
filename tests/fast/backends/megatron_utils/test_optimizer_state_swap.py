"""Two optimizer histories kept side by side on one model (optimizer_state_swap)."""

from types import SimpleNamespace

import pytest
import torch

from miles.backends.megatron_utils.optimizer_state_reset import UnsupportedOptimizerState
from miles.backends.megatron_utils.optimizer_state_swap import OptimizerStateStash


def _adam():
    param = torch.nn.Parameter(torch.ones(4))
    return param, torch.optim.AdamW([param], lr=0.1, weight_decay=0.0)


def _step(param, optimizer, grad):
    param.grad = torch.full_like(param, grad)
    optimizer.step()


def test_swap_keeps_each_history_and_starts_fresh_the_first_time():
    param, optimizer = _adam()
    stash = OptimizerStateStash()
    _step(param, optimizer, 1.0)  # phase A
    a_avg = optimizer.state[param]["exp_avg"].clone()
    out = stash.swap(optimizer, stash_as="a", load="b")
    assert out["load"] == "fresh" and not optimizer.state[param]["exp_avg"].any()
    _step(param, optimizer, -2.0)  # phase B from a fresh history
    b_avg = optimizer.state[param]["exp_avg"].clone()
    out = stash.swap(optimizer, stash_as="b", load="a")
    assert out["load"] == "restored"
    assert torch.equal(optimizer.state[param]["exp_avg"], a_avg) and int(optimizer.state[param]["step"]) == 1
    _step(param, optimizer, 1.0)
    assert int(optimizer.state[param]["step"]) == 2  # phase A continues its own history
    stash.swap(optimizer, stash_as="a", load="b")
    assert torch.equal(optimizer.state[param]["exp_avg"], b_avg) and int(optimizer.state[param]["step"]) == 1
    assert stash.names == ["a"]


def test_a_history_stashed_before_any_step_restores_fresh(tmp_path):
    """A new optimizer has no state before its first step (TE FusedAdam allocates lazily)."""
    param, optimizer = _adam()
    stash = OptimizerStateStash()
    stash.swap(optimizer, stash_as="a", load="b")  # A never stepped: an empty snapshot
    _step(param, optimizer, 1.0)
    out = stash.swap(optimizer, stash_as="b", load="a")
    assert out["load"] == "fresh" and not optimizer.state[param]["exp_avg"].any()
    b_avg = stash._snapshots["b"]["leaves"][0]["states"][0]["exp_avg"].clone()

    # a restart: the saved histories go into an optimizer that has never stepped
    stash.save(tmp_path / "stash.pt")
    restarted_param, restarted = _adam()
    reloaded = OptimizerStateStash()
    reloaded.load(tmp_path / "stash.pt")
    assert reloaded.names == ["b"]  # the empty history of A was consumed by the swap back
    reloaded.swap(restarted, stash_as="a", load="b")
    assert torch.equal(restarted.state[restarted_param]["exp_avg"], b_avg)
    assert float(restarted_param.detach().sum()) == 4.0  # the step that created the state moved nothing


def test_a_layout_mismatch_fails_before_writing():
    param, optimizer = _adam()
    _step(param, optimizer, 1.0)
    other_param, other = _adam()
    _step(other_param, other, 1.0)
    stash = OptimizerStateStash()
    stash.swap(other, stash_as="b", load="unused")
    stash._snapshots["b"]["leaves"][0]["states"][0]["exp_avg"] = torch.zeros(7)  # wrong shape
    before = optimizer.state[param]["exp_avg"].clone()
    with pytest.raises(RuntimeError, match="shape"):
        stash.swap(optimizer, stash_as="a", load="b")
    assert torch.equal(optimizer.state[param]["exp_avg"], before)


def test_exchange_keeps_a_single_host_copy():
    param, optimizer = _adam()
    stash = OptimizerStateStash()
    _step(param, optimizer, 1.0)
    stash.swap(optimizer, stash_as="a", load="b")
    _step(param, optimizer, -1.0)
    snapshot = stash._snapshots["a"]
    stash.swap(optimizer, stash_as="b", load="a")
    assert stash._snapshots["b"] is snapshot  # the same host buffers now hold the other history


def test_megatron_wrappers_are_descended():
    param, optimizer = _adam()
    _step(param, optimizer, 1.0)
    wrapped = SimpleNamespace(chained_optimizers=[SimpleNamespace(optimizer=optimizer)])
    out = OptimizerStateStash().swap(wrapped, stash_as="a", load="b")
    assert out["load"] == "fresh" and not optimizer.state[param]["exp_avg"].any()


def test_unknown_state_is_refused():
    param, optimizer = _adam()
    _step(param, optimizer, 1.0)
    optimizer.state[param]["unknown"] = torch.zeros(1)
    with pytest.raises(UnsupportedOptimizerState):
        OptimizerStateStash().swap(optimizer, stash_as="a", load="b")
