"""Synchronous SFT with teacher-forced validation and resumable latest/best saves.

Uses the same rollout adapter to render both training and validation examples.
There are no inference engines, rewards, policy snapshots, or stale samples.
"""

import asyncio
import copy
import json
import logging
import shutil
from pathlib import Path

from miles.ray.placement_group import create_rollout_components, create_training_models
from miles.ray.rollout.train_data_conversion import convert_samples_to_train_data, split_train_data_by_dp
from miles.utils.arguments import parse_args
from miles.utils.async_utils import Disposer, with_disposer
from miles.utils.data import Dataset, RolloutDataPack, remove_rollout_data_refs, remove_train_output_refs
from miles.utils.function_registry import load_function
from miles.utils.orchestration_utils import init_orchestration_script
from miles.utils.processing_utils import load_tokenizer
from miles.utils.tracking_utils import tracking

logger = logging.getLogger(__name__)


class _ValidationBuffer:
    def __init__(self, samples):
        self.samples = samples

    def get_samples(self, count):
        if count != len(self.samples):
            raise ValueError("Validation batch size mismatch")
        return [[copy.deepcopy(sample)] for sample in self.samples]


def _validation_batches(args):
    tokenizer = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
    dataset = Dataset(
        args.sft_validation_data,
        tokenizer=tokenizer,
        processor=None,
        prompt_key=args.input_key,
        label_key=None,
        max_length=None,
    )
    if not len(dataset):
        raise ValueError("Empty SFT validation dataset")
    render = load_function(args.rollout_function_path)
    size = args.global_batch_size
    for start in range(0, len(dataset), size):
        real = dataset.samples[start : start + size]
        # A short real row supplies legal tokens for padding, but contributes no loss.
        padding = min(real, key=lambda sample: len(str(sample.prompt)))
        rows = real + [padding] * (size - len(real))
        groups = render(args, 0, _ValidationBuffer(rows))
        samples = [group[0] for group in groups]
        for i, sample in enumerate(samples):
            sample.index = start + i
            sample.group_index = start + i
            if i >= len(real):
                sample.loss_mask = [0] * sample.response_length
        yield samples


async def _validate(args, actor, parallel, batches, step):
    nll = count = 0.0
    for samples in batches:
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
    logger.info("SFT validation step=%d loss=%.9f tokens=%d", step, loss, count)
    tracking.log(args, {"sft/step": step, "val/loss": loss, "val/tokens": count}, step_key="sft/step")
    return loss


def _read_state(root, iteration):
    if iteration < 0:
        return {"best_loss": None, "best_iteration": None}
    return json.loads((root / f"iter_{iteration:07d}" / "sft_state.json").read_text())


def _atomic_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def _finish_save(root, iteration, state, keep_best):
    """Prune only after the new model, optimizer and data cursor are durable."""
    checkpoint = root / f"iter_{iteration:07d}"
    if not (checkpoint / ".metadata").exists():
        raise RuntimeError(f"Incomplete distributed checkpoint: {checkpoint}")
    if int((root / "latest_checkpointed_iteration.txt").read_text()) != iteration:
        raise RuntimeError("Checkpoint tracker does not identify the completed save")
    if not (root / "rollout" / str(iteration)).is_dir():
        raise RuntimeError("Checkpoint is missing its data cursor")
    _atomic_json(checkpoint / "sft_state.json", state)
    keep = {iteration}
    if keep_best and state["best_iteration"] is not None:
        keep.add(state["best_iteration"])
    _atomic_json(root / "sft_best.json", state)
    for path in root.glob("iter_*"):
        suffix = path.name.removeprefix("iter_")
        if suffix.isdigit() and int(suffix) not in keep:
            shutil.rmtree(path)
            rollout = root / "rollout" / str(int(suffix))
            if rollout.is_dir():
                shutil.rmtree(rollout)


async def _train(args, *, disposer: Disposer):
    if not args.debug_train_only or args.loss_type != "sft_loss" or args.async_save:
        raise ValueError("train_sft requires --debug-train-only --loss-type sft_loss and synchronous saves")
    if args.global_batch_size != args.rollout_batch_size or args.n_samples_per_prompt != 1:
        raise ValueError("SFT requires one optimizer step per batch and one response per row")
    if args.save != args.load or args.save_retain_interval is not None:
        raise ValueError("Use one SFT load/save root and let this driver own checkpoint retention")
    if args.sft_eval_interval < 1 or args.save_interval is None or args.save_interval < 1:
        raise ValueError("SFT eval/save intervals must be positive")
    init_orchestration_script(args, disposer=disposer)
    controller, executor, _ = await create_rollout_components(args)
    disposer.add(controller, executor)
    actor, critic = await create_training_models(args, executor)
    disposer.add(critic, actor)
    parallel = await actor.get_train_parallel_config()
    epoch_steps = await executor.get_num_rollout_per_epoch()
    batches = list(_validation_batches(args))
    root = Path(args.save)
    state = _read_state(root, args.start_rollout_id - 1)
    if args.start_rollout_id == 0:
        await _validate(args, actor, parallel, batches, 0)
    for iteration in range(args.start_rollout_id, args.num_rollout):
        # No prefetch across a checkpoint: the persisted cursor matches these weights exactly.
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
            loss = await _validate(args, actor, parallel, batches, step)
            if state["best_loss"] is None or loss < state["best_loss"]:
                state = {"best_loss": loss, "best_iteration": iteration}
        if validate or step % args.save_interval == 0:
            await executor.save(iteration)
            checkpoint = root / f"iter_{iteration:07d}"
            checkpoint.mkdir(parents=True, exist_ok=True)
            _atomic_json(checkpoint / "sft_state.json", state)
            await actor.save_model(iteration, force_sync=True)
            _finish_save(root, iteration, state, args.sft_keep_best)
        if early_exit:
            break


def _arguments(parser):
    parser.add_argument("--sft-validation-data", required=True)
    parser.add_argument("--sft-eval-interval", type=int, default=100)
    parser.add_argument("--sft-keep-best", action="store_true")
    return parser


if __name__ == "__main__":
    asyncio.run(with_disposer(_train, parse_args(add_custom_arguments=_arguments)))
