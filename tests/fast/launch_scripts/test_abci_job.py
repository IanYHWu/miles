from pathlib import Path
from unittest.mock import Mock

import pytest
import torch
from tools.abci import launch


def test_failed_worker_stops_the_head_instead_of_waiting_for_unavailable_gpus(monkeypatch):
    head = Mock()
    head.poll.return_value = None
    worker = Mock(returncode=7)
    worker.poll.return_value = 7
    monkeypatch.setattr(launch.subprocess, "Popen", lambda command: head)
    with pytest.raises(RuntimeError, match="worker exited early"):
        launch._run_head(["unused"], [worker])
    head.terminate.assert_called_once()
    head.wait.assert_called_once_with(timeout=30)


def test_resume_checks_new_gradients_without_comparing_to_pretrained_weights(monkeypatch, tmp_path):
    args = launch._parser().parse_args(
        [
            "--mode",
            "smoke",
            "--image",
            "unused",
            "--repo",
            "unused",
            "--scratch",
            str(tmp_path),
            "--model-dir",
            "unused",
            "--data-dir",
            "unused",
            "--judge-config",
            "unused",
        ]
    )
    monkeypatch.setenv("PBS_JOBID", "123.pbs")
    run_dir = tmp_path / "runs" / "pbs-123-Qwen3.5-9B"
    calls = []

    def run(argv):
        calls.append(argv)
        checkpoint = run_dir / "checkpoints"
        checkpoint.mkdir(exist_ok=True)
        iteration = 1 if len(calls) == 1 else 2
        (checkpoint / "latest_checkpointed_iteration.txt").write_text(str(iteration))
        for step in range(iteration + 1):
            torch.save(1.5, run_dir / f"grad-{step}.pt")

    monkeypatch.setattr(launch, "_run", run)
    launch._execute_recipe(args)
    assert "--check-weight-update-equal" in calls[0][-1]
    assert "--check-weight-update-equal" not in calls[1][-1]
    assert Path(calls[1][calls[1].index("--load") + 1]) == run_dir / "checkpoints"
    assert "--num-rollout 3" in calls[1][-1]
