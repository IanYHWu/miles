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
    assert "--ci-test" in calls[0][-1] and "--ci-test" in calls[1][-1]
    assert "--num-rollout 3 --debug-exit-after-rollout 2" in calls[0][-1]
    assert "--debug-exit-after-rollout" not in calls[1][-1]
    assert Path(calls[1][calls[1].index("--load") + 1]) == run_dir / "checkpoints"
    assert "--num-rollout 3" in calls[1][-1]


def test_async_smoke_validates_in_a_fresh_interpreter(monkeypatch, tmp_path):
    args = launch._parser().parse_args(
        [
            "--mode",
            "async-smoke",
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
    monkeypatch.setenv("PBS_JOBID", "456.pbs")
    calls = []
    monkeypatch.setattr(launch, "_run", calls.append)
    launch._execute_recipe(args)
    assert len(calls) == 2
    assert "--async-smoke" in calls[0]
    assert "--custom-megatron-before-train-step-hook-path" in calls[0][-1]
    assert calls[1][1:3] == ["-m", "tests.manual.async_baseline_fixture"]


def test_train_forwards_rollout_topology_and_update_count(monkeypatch, tmp_path):
    args = launch._parser().parse_args([
        "--mode", "train", "--image", "unused", "--repo", "unused",
        "--scratch", str(tmp_path), "--model-dir", "unused", "--data-dir", "unused",
        "--judge-config", "unused", "--rollout-nodes", "3", "--num-rollout", "120",
    ])
    monkeypatch.setenv("PBS_JOBID", "789.pbs")
    calls = []
    monkeypatch.setattr(launch, "_run", calls.append)
    launch._execute_recipe(args)
    argv, = calls
    assert argv[argv.index("--rollout-gpus") + 1] == "24"
    assert argv[argv.index("--num-rollout") + 1] == "120"
    assert "--smoke" not in argv and "--extra-args" not in argv


def test_four_node_host_starts_all_workers_using_pbs_slot_indices(monkeypatch, tmp_path):
    nodefile = tmp_path / "nodes"
    nodefile.write_text("head\nhead\na\na\nb\nb\nc\nc\n")
    monkeypatch.setenv("PBS_JOBID", "789.pbs")
    monkeypatch.setenv("PBS_NODEFILE", str(nodefile))
    args = launch._parser().parse_args([
        "--mode", "train", "--image", str(tmp_path), "--repo", str(tmp_path),
        "--scratch", str(tmp_path), "--model-dir", str(tmp_path), "--data-dir", str(tmp_path),
        "--judge-config", str(tmp_path), "--rollout-nodes", "3", "--num-rollout", "120",
    ])
    commands, workers = [], []
    def popen(command):
        commands.append(command)
        worker = Mock()
        workers.append(worker)
        return worker
    monkeypatch.setattr(launch.socket, "gethostbyname", lambda _: "10.0.0.1")
    monkeypatch.setattr(launch.subprocess, "Popen", popen)
    head = Mock()
    monkeypatch.setattr(launch, "_run_head", head)
    launch._host(args)
    assert [command[2] for command in commands] == ["2", "4", "6"]
    assert len(head.call_args.args[1]) == 3
    for worker in workers:
        worker.terminate.assert_called_once()
