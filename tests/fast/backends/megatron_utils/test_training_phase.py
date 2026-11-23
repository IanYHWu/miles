"""In-place changes to a resident trainer's optimizer run (training_phase)."""

import torch
from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler

from miles.backends.megatron_utils.training_phase import start_new_optimizer_run


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
