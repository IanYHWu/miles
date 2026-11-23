"""Change what a resident Megatron trainer optimizes without rebuilding it.

The LR / weight-decay schedule lives in Megatron's ``OptimizerParamScheduler``, which writes each
param group's ``lr`` and ``weight_decay`` on every step; the optimizer history is reset with
``optimizer_state_reset``. The fp32 main params are never touched.
"""

from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler

from miles.backends.megatron_utils.optimizer_state_reset import reset_optimizer_states


def start_new_optimizer_run(optimizer, scheduler: OptimizerParamScheduler | None) -> dict:
    """Start a fresh optimizer run on the current fp32 main params, e.g. after loading a checkpoint with
    optimizer state: zero the history and step counters, and restart the LR schedule at step 0 with the
    scheduler's current settings (this run's arguments, unless the checkpoint's were loaded)."""
    reset_optimizer_states(optimizer)
    steps_before = None
    if scheduler is not None:
        steps_before = scheduler.num_steps
        scheduler.num_steps = 0
        scheduler.step(increment=0)
    return {"scheduler_steps_before": steps_before}


# Scheduler fields that param groups may override per group (Megatron's _OPT_PARAM_SCHEDULER_OVERRIDE_KEYS).
_PER_GROUP_FIELDS = ("max_lr", "min_lr", "start_wd", "end_wd")


def set_phase(
    args,
    optimizer,
    scheduler: OptimizerParamScheduler,
    *,
    args_overrides: dict,
    scheduler_fields: dict,
    reset_optimizer: bool,
) -> dict:
    """Switch a resident trainer to another objective and schedule on the same model and optimizer.

    ``args_overrides`` are set on the training args (e.g. ``loss_type``,
    ``compute_advantages_and_returns``); every key must already exist. ``scheduler_fields`` replace
    ``OptimizerParamScheduler`` attributes (``max_lr``, ``min_lr``, ``init_lr``, ``lr_warmup_steps``,
    ``lr_decay_steps``, ``lr_decay_style``, ``start_wd``, ``end_wd``, ``wd_incr_steps``,
    ``wd_incr_style``, ``num_steps``, ...); the per-group fields among them also replace a param
    group's own override, so every group follows the new schedule. ``reset_optimizer`` zeroes the
    optimizer history (a fresh optimizer on the current fp32 main params). Each param group's LR and
    weight decay are then re-derived at the scheduler's current step."""
    for key in args_overrides:
        if not hasattr(args, key):
            raise AttributeError(f"set_phase: unknown argument {key}")
    for key in scheduler_fields:
        if not hasattr(scheduler, key):
            raise AttributeError(f"set_phase: unknown scheduler field {key}")
    for key, value in args_overrides.items():
        setattr(args, key, value)
    for key, value in scheduler_fields.items():
        setattr(scheduler, key, value)
        if key in _PER_GROUP_FIELDS:
            for group in optimizer.param_groups:
                if key in group:
                    group[key] = value
    if reset_optimizer:
        reset_optimizer_states(optimizer)
    scheduler.step(increment=0)
    return {
        "lr": sorted({float(group["lr"]) for group in optimizer.param_groups}),
        "weight_decay": sorted({float(group["weight_decay"]) for group in optimizer.param_groups}),
        "num_steps": scheduler.num_steps,
        "reset_optimizer": reset_optimizer,
    }
