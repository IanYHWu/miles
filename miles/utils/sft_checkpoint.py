"""Commit model, optimizer, data cursor, and SFT metrics as one resumable step.

Megatron publishes its own tracker during model saving. The SFT manifest is the
commit point: an interrupted save is rolled back to it before workers start.
The save directory must be owned exclusively by one SFT driver.
"""

import json
import os
import shutil
from pathlib import Path

_MANIFEST = "sft_latest.json"
_PENDING = "sft_pending.json"
_TRACKER = "latest_checkpointed_iteration.txt"


def _atomic_write(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_json(path: Path, data: dict) -> None:
    _atomic_write(path, json.dumps(data, indent=2) + "\n")


def _checkpoint(root: Path, iteration: int) -> Path:
    return root / f"iter_{iteration:07d}"


def _require_complete(root: Path, iteration: int) -> dict:
    checkpoint = _checkpoint(root, iteration)
    if not (checkpoint / ".metadata").is_file():
        raise RuntimeError(f"Incomplete distributed checkpoint: {checkpoint}")
    # The executor publishes this directory atomically after DataSource.save.
    # The data source owns its serialization format, including filenames.
    if not (root / "rollout" / str(iteration) / "data_source").is_dir():
        raise RuntimeError(f"Checkpoint {iteration} is missing its data cursor")
    return json.loads((checkpoint / "sft_state.json").read_text())


def recover_checkpoint(root: Path) -> dict:
    """Recover a previously committed step before Megatron reads its tracker."""
    root.mkdir(parents=True, exist_ok=True)
    manifest = root / _MANIFEST
    pending = root / _PENDING
    tracker = root / _TRACKER
    if manifest.exists():
        commit = json.loads(manifest.read_text())
        iteration = commit["iteration"]
        state = _require_complete(root, iteration)
        if state != commit["state"]:
            raise RuntimeError("SFT manifest and checkpoint state disagree")
        _atomic_write(tracker, str(iteration))
    else:
        if tracker.exists() and not pending.exists():
            raise ValueError("SFT load/save root contains a checkpoint without an SFT commit manifest")
        tracker.unlink(missing_ok=True)
        commit = dict(iteration=-1, state=dict(best_loss=None, best_iteration=None))
    if pending.exists():
        candidate = json.loads(pending.read_text())["iteration"]
        if candidate != commit["iteration"]:
            # This directory belongs to this driver's interrupted transaction.
            shutil.rmtree(_checkpoint(root, candidate), ignore_errors=True)
            shutil.rmtree(root / "rollout" / str(candidate), ignore_errors=True)
        pending.unlink()
    return commit


def begin_save(root: Path, iteration: int, state: dict) -> None:
    """Stage bookkeeping before the model saver can publish its tracker."""
    checkpoint = _checkpoint(root, iteration)
    if checkpoint.exists():
        raise FileExistsError(f"Refusing to overwrite checkpoint {checkpoint}")
    _atomic_json(root / _PENDING, dict(iteration=iteration))
    checkpoint.mkdir()
    _atomic_json(checkpoint / "sft_state.json", state)


def finish_save(root: Path, iteration: int, state: dict, *, keep_best: bool) -> None:
    """Publish the commit, then prune only completed checkpoints owned by SFT."""
    saved_state = _require_complete(root, iteration)
    if saved_state != state:
        raise RuntimeError("Staged SFT state changed while saving")
    if int((root / _TRACKER).read_text()) != iteration:
        raise RuntimeError("Checkpoint tracker does not identify the completed save")
    keep = {iteration}
    if keep_best and state["best_iteration"] is not None:
        _require_complete(root, state["best_iteration"])
        keep.add(state["best_iteration"])
    _atomic_json(root / _MANIFEST, dict(iteration=iteration, state=state))
    (root / _PENDING).unlink(missing_ok=True)
    for checkpoint in root.glob("iter_*"):
        suffix = checkpoint.name.removeprefix("iter_")
        if suffix.isdigit() and int(suffix) not in keep and (checkpoint / "sft_state.json").exists():
            shutil.rmtree(checkpoint)
            shutil.rmtree(root / "rollout" / str(int(suffix)), ignore_errors=True)
