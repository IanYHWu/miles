"""A full fine-tuning window reduces its gradients once, steps under the request's AdamParams, and never steps on a
non-finite gradient."""

import math
from types import SimpleNamespace

import pytest
import torch

from miles.backends.megatron_utils.full_finetune import model as full_model
from miles.backends.megatron_utils.full_finetune.model import GradientWindow

ADAM = {
    "learning_rate": 1e-6,
    "beta1": 0.9,
    "beta2": 0.95,
    "eps": 1e-8,
    "weight_decay": 0.1,
    "grad_clip_norm": 1.0,
}


class _FakeOptimizer:
    """Duck ChainedOptimizer: records the step choreography without a Megatron build."""

    def __init__(self, *, grad_norm: float = 2.0, found_inf: bool = False) -> None:
        self.chained_optimizers = [SimpleNamespace(config=SimpleNamespace(clip_grad=0.0)) for _ in range(2)]
        self.param_groups = [
            {"lr": torch.tensor(0.0), "wd_mult": 1.0},
            {"lr": 0.0, "wd_mult": 0.0},
        ]
        self._grad_norm = grad_norm
        self._found_inf = found_inf
        self.calls: list[str] = []

    def prepare_grads(self) -> bool:
        self.calls.append("prepare_grads")
        return self._found_inf

    def get_grad_norm(self) -> float:
        self.calls.append("get_grad_norm")
        return self._grad_norm

    def step(self):
        self.calls.append("step")
        return True, self._grad_norm, None

    def zero_grad(self) -> None:
        self.calls.append("zero_grad")


class _FakeChunk:
    def __init__(self) -> None:
        self.zeroed = 0

    def zero_grad_buffer(self) -> None:
        self.zeroed += 1


@pytest.fixture
def reductions(monkeypatch):
    calls = []
    monkeypatch.setattr(full_model, "finalize_model_grads_with_empty_cache", lambda *args: calls.append(args))
    monkeypatch.setattr(full_model.ModelCompanionWeightVersionUtils, "bump_weight_version", lambda model: None)
    return calls


def test_a_window_reduces_its_gradients_once_at_the_step(reductions):
    model, optimizer, window = [_FakeChunk()], _FakeOptimizer(), GradientWindow()
    window.defer_reduction(model, None)
    window.defer_reduction(model, None)

    full_model.optim_step(model, optimizer, window, ADAM)
    assert reductions == [(model, None)]

    full_model.optim_step(model, optimizer, window, ADAM)
    assert len(reductions) == 1, "a step with no backward since the last has nothing to reduce"


def test_adam_params_reach_every_group_with_megatrons_weight_decay_multiplier(reductions):
    optimizer = _FakeOptimizer()

    full_model.optim_step([_FakeChunk()], optimizer, GradientWindow(), ADAM)

    decayed, undecayed = optimizer.param_groups
    assert decayed["lr"].item() == pytest.approx(1e-6) and decayed["weight_decay"] == pytest.approx(0.1)
    assert undecayed["lr"] == pytest.approx(1e-6) and undecayed["weight_decay"] == 0.0
    assert all(group["betas"] == (0.9, 0.95) and group["eps"] == 1e-8 for group in optimizer.param_groups)
    assert [child.config.clip_grad for child in optimizer.chained_optimizers] == [1.0, 1.0]


def test_a_finite_step_reports_its_norm_and_drops_the_window(reductions):
    model, optimizer = [_FakeChunk()], _FakeOptimizer(grad_norm=2.0)

    outcome = full_model.optim_step(model, optimizer, GradientWindow(), ADAM)

    assert outcome == {"grad_norm": 2.0}
    assert optimizer.calls == ["prepare_grads", "get_grad_norm", "step", "zero_grad"]
    assert model[0].zeroed == 1


@pytest.mark.parametrize(
    "optimizer",
    [_FakeOptimizer(grad_norm=math.nan), _FakeOptimizer(grad_norm=math.inf), _FakeOptimizer(found_inf=True)],
    ids=["nan-norm", "inf-norm", "found-inf"],
)
def test_a_non_finite_gradient_never_reaches_the_weights(reductions, optimizer):
    model, window = [_FakeChunk()], GradientWindow()
    window.defer_reduction(model, None)

    outcome = full_model.optim_step(model, optimizer, window, ADAM)

    assert outcome == {"skipped_nonfinite": 1.0}
    assert "step" not in optimizer.calls
    assert optimizer.calls[-1] == "zero_grad" and model[0].zeroed == 1
    assert not window.reduction_pending
