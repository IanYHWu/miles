"""Synchronous Megatron SFT with teacher-forced validation and resumable saves.

The rollout callable renders gold responses; no inference engines are launched.
Training and validation use the same dataset schema and rendering callable.
"""

import asyncio
import copy
import inspect
import logging
import math
from pathlib import Path

from miles.backends.megatron_utils.megatron_config import resolve_args_checkpoint_load
from miles.ray.placement_group import create_rollout_components, create_training_models
from miles.ray.rollout.train_data_conversion import convert_samples_to_train_data, split_train_data_by_dp
from miles.rollout.base_types import RolloutFnTrainOutput
from miles.utils.arguments import parse_args
from miles.utils.async_utils import Disposer, with_disposer
from miles.utils.data import RolloutDataPack, remove_rollout_data_refs, remove_train_output_refs
from miles.utils.environ import use_legacy_rollout_v1
from miles.utils.function_registry import load_function
from miles.utils.orchestration_utils import init_orchestration_script
from miles.utils.sft_checkpoint import begin_save, finish_save, recover_checkpoint
from miles.utils.tracking_utils import tracking

logger = logging.getLogger(__name__)


class _ValidationBuffer:
    def __init__(self, samples):
        self.samples = samples

    def get_samples(self, count):
        if count != len(self.samples):
            raise ValueError("Validation batch size mismatch")
        return [[copy.deepcopy(sample)] for sample in self.samples]


def _validation_dataset(args):
    validation_args = copy.copy(args)
    validation_args.prompt_data = args.sft_validation_data
    validation_args.rollout_shuffle = False
    validation_args.dump_details = None
    source = load_function(args.sft_validation_data_source_path)(validation_args)
    if not hasattr(source, "dataset") or source.dataset is None or not hasattr(source.dataset, "samples"):
        raise ValueError("SFT validation data source must expose a finite dataset with samples")
    return source.dataset


async def _validation_batches(args, dataset, render):
    """Render one batch at a time; never retain the tokenized validation corpus."""
    if not len(dataset):
        raise ValueError("Empty SFT validation dataset")
    size = args.global_batch_size
    for start in range(0, len(dataset), size):
        real = dataset.samples[start : start + size]
        # Pad the collective schedule, but remove padded rows from the objective.
        rows = real + [real[0]] * (size - len(real))
        output = render(args, 0, _ValidationBuffer(rows), evaluation=False)
        if inspect.isawaitable(output):
            output = await output
        if isinstance(output, RolloutFnTrainOutput):
            output = output.samples
        if len(output) != size or any(len(row) != 1 for row in output):
            raise ValueError("SFT renderer must return one response per input, in input order")
        samples = [row[0] for row in output]
        for i, sample in enumerate(samples):
            sample.index = start + i
            sample.group_index = start + i
            if sample.loss_mask is None or len(sample.loss_mask) != sample.response_length:
                raise ValueError("SFT renderer must provide a response-aligned loss mask")
            if i >= len(real):
                sample.loss_mask = [0] * sample.response_length
        yield samples


async def _validate(args, actor, parallel, dataset, render, step):
    nll = count = 0.0
    async for samples in _validation_batches(args, dataset, render):
        data = convert_samples_to_train_data(args, samples, {}, None, None)
        pack = RolloutDataPack(data_ref=split_train_data_by_dp(args, data, parallel))
        try:
            result = await actor.evaluate_sft(step - 1, pack)
            nll += result["nll_sum"]
            count += result["token_count"]
        finally:
            remove_rollout_data_refs(args, pack)
    if count <= 0:
        raise ValueError("Validation has no supervised tokens")
    loss = nll / count
    if not math.isfinite(loss):
        raise ValueError("Validation produced a non-finite loss")
    logger.info("SFT validation step=%d loss=%.9f tokens=%d", step, loss, count)
    tracking.log(args, {"sft/step": step, "val/loss": loss, "val/tokens": count}, step_key="sft/step")
    return loss


def _validate_configuration(args):
    if (
        args.train_backend != "megatron"
        or args.ckpt_format != "torch_dist"
        or args.indep_dp
        or args.megatron_config is not None
    ):
        raise ValueError("train_sft requires one Megatron training cell and torch_dist checkpoints")
    if (
        not args.debug_train_only
        or args.loss_type != "sft_loss"
        or args.async_save
        or args.offload_train
        or args.eval_num_gpus
    ):
        raise ValueError("train_sft requires resident --debug-train-only SFT models and synchronous saves")
    if args.use_critic or args.fully_async:
        raise ValueError("SFT does not use a critic or an asynchronous rollout queue")
    if args.global_batch_size != args.rollout_batch_size or args.n_samples_per_prompt != 1:
        raise ValueError("SFT requires one optimizer step per batch and one response per row")
    if not args.save or not args.requested_load or Path(args.save).resolve() != Path(args.requested_load).resolve():
        raise ValueError("Use one nonempty SFT load/save root")
    if args.save_retain_interval is not None or args.sft_eval_interval < 1 or not args.save_interval:
        raise ValueError("Use positive SFT eval/save intervals and let this driver own retention")
    if args.save_interval < 1:
        raise ValueError("SFT save interval must be positive")
    if args.no_save_optim or args.no_save_rng:
        raise ValueError("SFT checkpoints require optimizer and RNG state")
    if args.start_rollout_id not in (None, 0) or args.ckpt_step is not None:
        raise ValueError("SFT resume is selected by its committed checkpoint, not a manual step override")
    if not args.rollout_global_dataset:
        raise ValueError("SFT requires a resumable global dataset")
    if args.sft_validation_data_source_path is None:
        if args.data_source_path != "miles.rollout.data_source.RolloutDataSource":
            raise ValueError("Custom training data sources require explicit --sft-validation-data-source-path")
        args.sft_validation_data_source_path = args.data_source_path
    if (
        args.use_dynamic_global_batch_size
        or args.custom_convert_samples_to_train_data_path
        or args.custom_reward_post_process_path
    ):
        raise ValueError("SFT requires fixed batches and the standard sample-to-training-data conversion")
    if args.qkv_format not in ("thd", "bshd") or args.allgather_cp:
        raise ValueError("SFT validation supports THD/BSHD with standard context parallelism, not --allgather-cp")
    render = load_function(args.rollout_function_path)
    if inspect.isclass(render):
        raise ValueError("train_sft requires a stateless rendering callable, not a stateful rollout class")
    if inspect.iscoroutinefunction(render) and not use_legacy_rollout_v1():
        raise ValueError("Async legacy SFT renderers require MILES_USE_LEGACY_ROLLOUT_V1=1")
    return render


async def _save(args, actor, executor, root, iteration, state):
    begin_save(root, iteration, state)
    await executor.save(iteration)
    await actor.save_model(iteration, force_sync=True)
    finish_save(root, iteration, state, keep_best=args.sft_keep_best)


async def _train(args, *, disposer: Disposer):
    render = _validate_configuration(args)
    root = Path(args.save)
    commit = recover_checkpoint(root)
    if commit["iteration"] >= 0:
        if args.start_rollout_id == 0 and args.load != args.requested_load:
            # A missing tracker made parse_args choose fresh initialization. The
            # recovered SFT manifest wins: always restore its complete state,
            # even if no-load flags were supplied along with that fallback.
            args.start_rollout_id = None
            args.no_load_optim = args.no_load_rng = args.finetune = False
        if args.no_load_optim or args.no_load_rng or args.finetune or args.start_rollout_id is not None:
            raise ValueError("SFT resume must restore optimizer/RNG and infer its step from the checkpoint")
    # parse_args resolves fresh initialization before recovery. Re-run that owner
    # after repairing an interrupted save so trainer and cursor choose one step.
    args.load = args.requested_load
    resolve_args_checkpoint_load(args)
    state = commit["state"]
    if args.sft_keep_best and state["best_iteration"] is not None:
        if not (root / f"iter_{state['best_iteration']:07d}" / ".metadata").is_file():
            raise ValueError("The retained best checkpoint is missing")
    dataset = _validation_dataset(args)
    if not len(dataset):
        raise ValueError("Empty SFT validation dataset")
    init_orchestration_script(args, disposer=disposer)
    controller, executor, _ = await create_rollout_components(args)
    disposer.add(controller, executor)
    actor, critic = await create_training_models(args, executor)
    disposer.add(critic, actor)
    if args.start_rollout_id != commit["iteration"] + 1:
        raise RuntimeError("Trainer and committed SFT data cursor disagree")
    parallel = await actor.get_train_parallel_config()
    epoch_steps = await executor.get_num_rollout_per_epoch()
    if epoch_steps < 1:
        raise ValueError("Training dataset must contain at least one global batch")
    if args.start_rollout_id == 0:
        await _validate(args, actor, parallel, dataset, render, 0)
    for iteration in range(args.start_rollout_id, args.num_rollout):
        # No prefetch across a save: cursor and weights refer to the same batch.
        pack = await executor.get(iteration)
        try:
            output = await actor.train(iteration, pack)
            remove_train_output_refs(output)
        finally:
            remove_rollout_data_refs(args, pack)
        step = iteration + 1
        epoch_end = step % epoch_steps == 0
        early_exit = bool(
            args.debug_exit_after_rollout and step - args.start_rollout_id >= args.debug_exit_after_rollout
        )
        final = step == args.num_rollout or early_exit
        tracking.log(
            args,
            {
                "sft/step": step,
                "epoch/current": (step - 1) // epoch_steps + 1,
                "epoch/step": (step - 1) % epoch_steps + 1,
            },
            step_key="sft/step",
        )
        validate = step % args.sft_eval_interval == 0 or epoch_end or final
        if validate:
            loss = await _validate(args, actor, parallel, dataset, render, step)
            if state["best_loss"] is None or loss < state["best_loss"]:
                state = dict(best_loss=loss, best_iteration=iteration)
        if validate or step % args.save_interval == 0:
            if not args.sft_keep_best and state["best_iteration"] != iteration:
                state = dict(best_loss=state["best_loss"], best_iteration=None)
            await _save(args, actor, executor, root, iteration, state)
        if early_exit:
            break


def _arguments(parser):
    parser.add_argument("--sft-validation-data", required=True)
    parser.add_argument(
        "--sft-validation-data-source-path",
        help="DataSource exposing a dataset with len()/samples; required for custom training data sources",
    )
    parser.add_argument("--sft-eval-interval", type=int, default=100)
    parser.add_argument(
        "--sft-keep-best", action="store_true", help="Retain best validated checkpoint in addition to latest"
    )
    return parser


if __name__ == "__main__":
    asyncio.run(with_disposer(_train, parse_args(add_custom_arguments=_arguments)))
