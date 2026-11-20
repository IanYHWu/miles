"""The Tinker gateway's trainer under ``--tinker-train-mode full``: one model at a time trains the served weights.

It answers the slot commands the multi-LoRA actor answers, for slot 0, so the gateway drives both alike. A sampler
save is a weight push, which the gateway makes through ``update_weights``.
"""

from contextlib import ExitStack

from miles.backends.megatron_utils.actor import MegatronTrainRayActor
from miles.backends.megatron_utils.full_finetune import model as full_model
from miles.backends.megatron_utils.full_finetune.model import GradientWindow
from miles.backends.megatron_utils.optimizer_utils import reset_optimizer_state
from miles.backends.megatron_utils.work_unit import run_forward_backward
from miles.backends.training_utils.data.rollout import get_rollout_data
from miles.utils.object_store import StoreObjectRef
from miles.utils.tracking_utils.structured_log import with_logs


class FullFinetuneTrainRayActor(MegatronTrainRayActor):
    def _init_training_state(self) -> None:
        super()._init_training_state()
        self.gradient_window = GradientWindow()
        # the weights the trainer loaded at startup, which a new model starts from
        self._holds_initial_weights = True

    @with_logs
    def forward_backward(self, batch_id: int, rollout_data_ref: StoreObjectRef) -> dict:
        self._heartbeat.bump()
        with ExitStack() as stack:
            rollout_data, store_get_result = get_rollout_data(self.args, rollout_data_ref)
            stack.enter_context(store_get_result)
            return run_forward_backward(
                self.args,
                batch_id,
                self.model,
                rollout_data,
                finalize_model_grads_func=self.gradient_window.defer_reduction,
            )

    @with_logs
    def optim_step(self, adam_params_by_slot: dict[int, dict]) -> dict[int, dict]:
        self._heartbeat.bump()
        [(slot, adam_params)] = adam_params_by_slot.items()
        outcome = full_model.optim_step(self.model, self.optimizer, self.gradient_window, adam_params)
        if "grad_norm" in outcome:
            self._holds_initial_weights = False
        return {slot: outcome}

    @with_logs
    def forward_only(self, batch_id: int, rollout_data_ref: StoreObjectRef) -> dict:
        """Same loss pass as forward_backward, without the backward: the Tinker
        forward() contract returns the requested loss per datum."""
        self._heartbeat.bump()
        with ExitStack() as stack:
            rollout_data, store_get_result = get_rollout_data(self.args, rollout_data_ref)
            stack.enter_context(store_get_result)
            return run_forward_backward(self.args, batch_id, self.model, rollout_data, forward_only=True)

    @with_logs
    def load_slot(
        self, slot: int, rank: int, alpha: float, ckpt_path: str | None = None, load_optimizer: bool = True
    ) -> None:
        """A new model starts from the initial weights with a fresh optimizer, or from a save_slot checkpoint.

        ``rank`` and ``alpha`` describe adapters, which this actor has none of.
        """
        self._heartbeat.bump()
        full_model.discard_gradients(self.model, self.optimizer, self.gradient_window)
        if ckpt_path is not None:
            self._load_checkpoint(ckpt_path, load_optimizer=load_optimizer)
        elif not self._holds_initial_weights:
            self._reload_initial_weights()

    @with_logs
    def save_slot(self, slot: int, path: str, metadata: dict | None = None) -> None:
        self._heartbeat.bump()
        full_model.save_checkpoint(self.args, self.model, self.optimizer, path, metadata)

    @with_logs
    def unload_slot(self, slot: int) -> dict | None:
        full_model.discard_gradients(self.model, self.optimizer, self.gradient_window)
        return None

    def _load_checkpoint(self, path: str, *, load_optimizer: bool) -> None:
        overrides = dict(load=path, ckpt_step=None, finetune=False, no_load_optim=not load_optimizer, no_load_rng=True)
        self._load_state_core(checkpointing_context=None, overrider_for_loading=overrides)
        if not load_optimizer:
            self._reset_optimizer_state()
        self._holds_initial_weights = False

    def _reload_initial_weights(self) -> None:
        overrides = dict(finetune=True, no_load_optim=True, no_load_rng=True)
        self._load_state_core(checkpointing_context=None, overrider_for_loading=overrides)
        self._reset_optimizer_state()
        self._holds_initial_weights = True

    def _reset_optimizer_state(self) -> None:
        reset_optimizer_state(
            self.optimizer,
            stream_optimizer_state_to_disk=self.args.stream_optimizer_state_to_disk,
            chunked_optimizer_state_offload=self.args.chunked_optimizer_state_offload,
        )
