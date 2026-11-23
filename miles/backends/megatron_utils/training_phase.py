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
