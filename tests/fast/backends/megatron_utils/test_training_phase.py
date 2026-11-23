"""In-place changes to a resident trainer's optimizer run (training_phase)."""

from types import SimpleNamespace

import pytest
import torch
from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler

from miles.backends.megatron_utils.training_phase import set_phase, start_new_optimizer_run


def _trainer(lr=1e-3, decay_style="cosine"):
    param = torch.nn.Parameter(torch.ones(4))
    optimizer = torch.optim.AdamW([param], lr=lr, weight_decay=0.0)
    scheduler = OptimizerParamScheduler(
        optimizer,
        init_lr=0.0,
        max_lr=lr,
        min_lr=lr / 10,
        lr_warmup_steps=2,
        lr_decay_steps=10,
        lr_decay_style=decay_style,
        start_wd=0.1,
        end_wd=0.1,
        wd_incr_steps=10,
        wd_incr_style="constant",
    )
    return param, optimizer, scheduler


def _step(param, optimizer, scheduler):
    param.grad = torch.ones_like(param)
    optimizer.step()
    scheduler.step(increment=1)


def test_a_new_run_keeps_the_weights_and_restarts_history_and_schedule():
    param, optimizer, scheduler = _trainer()
    for _ in range(3):
        _step(param, optimizer, scheduler)
    weights = param.detach().clone()

    out = start_new_optimizer_run(optimizer, scheduler)

    assert out == {"scheduler_steps_before": 3}
    assert scheduler.num_steps == 0 and optimizer.param_groups[0]["lr"] == 0.0  # warmup starts at init_lr
    state = optimizer.state[param]
    assert not state["exp_avg"].any() and not state["exp_avg_sq"].any() and int(state["step"]) == 0
    assert torch.equal(param.detach(), weights)


def test_a_new_run_without_a_scheduler_only_resets_the_history():
    param, optimizer, _ = _trainer()
    param.grad = torch.ones_like(param)
    optimizer.step()
    assert start_new_optimizer_run(optimizer, None) == {"scheduler_steps_before": None}
    assert not optimizer.state[param]["exp_avg"].any()


SFT = {
    "max_lr": 2e-6,
    "min_lr": 2e-7,
    "init_lr": 0.0,
    "lr_warmup_steps": 2,
    "lr_decay_steps": 10,
    "lr_decay_style": "cosine",
    "start_wd": 0.1,
    "end_wd": 0.1,
    "wd_incr_steps": 10,
    "wd_incr_style": "constant",
    "num_steps": 0,
}


def test_set_phase_switches_objective_schedule_and_history_in_place():
    param, optimizer, scheduler = _trainer(lr=1e-6, decay_style="constant")
    _step(param, optimizer, scheduler)
    args = SimpleNamespace(loss_type="policy_loss", compute_advantages_and_returns=True)
    weights = param.detach().clone()

    out = set_phase(
        args,
        optimizer,
        scheduler,
        args_overrides={"loss_type": "sft_loss", "compute_advantages_and_returns": False},
        scheduler_fields=dict(SFT, num_steps=1),
        reset_optimizer=True,
    )

    assert args.loss_type == "sft_loss" and args.compute_advantages_and_returns is False
    assert out == {"lr": [pytest.approx(1e-6)], "weight_decay": [0.1], "num_steps": 1, "reset_optimizer": True}
    assert not optimizer.state[param]["exp_avg"].any() and torch.equal(param.detach(), weights)
    scheduler.step(increment=1)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(2e-6)  # end of the SFT warmup


def test_set_phase_overrides_per_group_scheduler_fields():
    _, optimizer, scheduler = _trainer()
    optimizer.param_groups[0]["max_lr"] = 5.0  # a group override takes precedence in get_lr
    set_phase(
        SimpleNamespace(),
        optimizer,
        scheduler,
        args_overrides={},
        scheduler_fields={"max_lr": 3e-4, "min_lr": 3e-4, "lr_warmup_steps": 0, "lr_decay_style": "constant"},
        reset_optimizer=False,
    )
    assert optimizer.param_groups[0]["max_lr"] == 3e-4 and optimizer.param_groups[0]["lr"] == pytest.approx(3e-4)


@pytest.mark.parametrize(
    "overrides,fields", [({"no_such_arg": 1}, {}), ({}, {"no_such_field": 1})], ids=["argument", "scheduler"]
)
def test_set_phase_rejects_unknown_names_before_changing_anything(overrides, fields):
    _, optimizer, scheduler = _trainer()
    args = SimpleNamespace(loss_type="policy_loss")
    with pytest.raises(AttributeError, match="unknown"):
        set_phase(
            args,
            optimizer,
            scheduler,
            args_overrides=dict(overrides, loss_type="sft_loss"),
            scheduler_fields=dict(fields, max_lr=1.0),
            reset_optimizer=False,
        )
    assert args.loss_type == "policy_loss" and scheduler.max_lr == 1e-3
