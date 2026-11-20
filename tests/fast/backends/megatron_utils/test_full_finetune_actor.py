"""A new full fine-tuning model starts from the initial weights with a fresh optimizer, reloading them only when
something has changed them; a weights-only checkpoint load also starts a fresh optimizer."""

from argparse import Namespace
from types import SimpleNamespace

import pytest

from miles.backends.megatron_utils.full_finetune import actor as actor_module
from miles.backends.megatron_utils.full_finetune.actor import FullFinetuneTrainRayActor
from miles.backends.megatron_utils.full_finetune.model import GradientWindow


@pytest.fixture
def actor(monkeypatch):
    calls = []
    actor = FullFinetuneTrainRayActor.__new__(FullFinetuneTrainRayActor)
    actor.args = Namespace(stream_optimizer_state_to_disk=False, chunked_optimizer_state_offload=False)
    actor.model, actor.optimizer = ["chunk"], "optimizer"
    actor.gradient_window = GradientWindow()
    actor._holds_initial_weights = True
    actor._heartbeat = SimpleNamespace(bump=lambda: None)
    actor._load_state_core = lambda **kwargs: calls.append(("load", kwargs["overrider_for_loading"]))
    monkeypatch.setattr(actor_module.full_model, "discard_gradients", lambda *_args: calls.append(("discard",)))
    monkeypatch.setattr(actor_module, "reset_optimizer_state", lambda *_args, **_kwargs: calls.append(("reset",)))
    actor.calls = calls
    return actor


def test_a_new_model_on_the_initial_weights_reloads_nothing(actor):
    actor.load_slot(0, 0, 0.0)

    assert actor.calls == [("discard",)]


def test_a_new_model_after_training_reloads_the_initial_weights_with_a_fresh_optimizer(actor):
    actor._holds_initial_weights = False

    actor.load_slot(0, 0, 0.0)

    assert actor.calls == [
        ("discard",),
        ("load", {"finetune": True, "no_load_optim": True, "no_load_rng": True}),
        ("reset",),
    ]
    assert actor._holds_initial_weights


@pytest.mark.parametrize("load_optimizer", [True, False], ids=["with-optimizer", "weights-only"])
def test_a_checkpoint_load_keeps_its_optimizer_state_only_when_asked(actor, load_optimizer):
    actor.load_slot(0, 0, 0.0, ckpt_path="/checkpoints/ck", load_optimizer=load_optimizer)

    overrides = {
        "load": "/checkpoints/ck",
        "ckpt_step": None,
        "finetune": False,
        "no_load_optim": not load_optimizer,
        "no_load_rng": True,
    }
    assert actor.calls == [("discard",), ("load", overrides)] + ([] if load_optimizer else [("reset",)])
    assert not actor._holds_initial_weights


@pytest.mark.parametrize(
    "outcome,still_initial",
    [({"grad_norm": 1.0}, False), ({"skipped_nonfinite": 1.0}, True)],
    ids=["stepped", "skipped"],
)
def test_only_a_step_taken_changes_the_initial_weights(actor, monkeypatch, outcome, still_initial):
    monkeypatch.setattr(actor_module.full_model, "optim_step", lambda *_args: outcome)

    assert actor.optim_step({0: {"learning_rate": 1e-6}}) == {0: outcome}
    assert actor._holds_initial_weights is still_initial
