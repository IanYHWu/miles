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
    assert calls[1][1:3] == ["-m", "tests.manual.async_smoke_fixture"]


def test_train_forwards_rollout_topology_and_update_count(monkeypatch, tmp_path):
    args = launch._parser().parse_args(
        [
            "--mode",
            "train",
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
            "--rollout-nodes",
            "3",
            "--num-rollout",
            "120",
            "--load",
            "/existing/checkpoints",
            "--checkpoint-dir",
            "/existing/checkpoints",
            "--eval-before-train",
            "--save-interval",
            "5",
        ]
    )
    monkeypatch.setenv("PBS_JOBID", "789.pbs")
    calls = []
    monkeypatch.setattr(launch, "_run", calls.append)
    launch._execute_recipe(args)
    (argv,) = calls
    assert argv[argv.index("--rollout-gpus") + 1] == "24"
    assert argv[argv.index("--num-rollout") + 1] == "120"
    assert argv[argv.index("--load") + 1] == argv[argv.index("--checkpoint-dir") + 1] == "/existing/checkpoints"
    assert "--smoke" not in argv and "--extra-args" not in argv
    assert "--no-skip-eval-before-train" in argv
    assert argv[argv.index("--save-interval") + 1] == "5"


@pytest.mark.parametrize("actor_nodes,rollout_nodes", [(1, 3), (4, 4)])
def test_host_starts_all_workers_using_pbs_slot_indices(monkeypatch, tmp_path, actor_nodes, rollout_nodes):
    nodefile = tmp_path / "nodes"
    nodefile.write_text("".join(f"node{i}\nnode{i}\n" for i in range(actor_nodes + rollout_nodes)))
    monkeypatch.setenv("PBS_JOBID", "789.pbs")
    monkeypatch.setenv("PBS_NODEFILE", str(nodefile))
    args = launch._parser().parse_args(
        [
            "--mode",
            "train",
            "--image",
            str(tmp_path),
            "--repo",
            str(tmp_path),
            "--scratch",
            str(tmp_path),
            "--model-dir",
            str(tmp_path),
            "--data-dir",
            str(tmp_path),
            "--judge-config",
            str(tmp_path),
            "--actor-nodes",
            str(actor_nodes),
            "--rollout-nodes",
            str(rollout_nodes),
            "--checkpoint-dir",
            str(tmp_path / "group-checkpoints"),
            "--num-rollout",
            "120",
        ]
    )
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
    assert [command[2] for command in commands] == [str(2 * i) for i in range(1, actor_nodes + rollout_nodes)]
    assert len(head.call_args.args[1]) == actor_nodes + rollout_nodes - 1
    assert f"{tmp_path}/group-checkpoints:{tmp_path}/group-checkpoints:rw" in head.call_args.args[0]
    assert "/groups:/groups:ro" in head.call_args.args[0]
    for worker in workers:
        worker.terminate.assert_called_once()


@pytest.mark.parametrize("initial_failure", [False, True])
def test_production_resume_verification_requires_published_checkpoint(monkeypatch, tmp_path, initial_failure):
    from types import SimpleNamespace

    args = SimpleNamespace(
        verify_resume_after_updates=5,
        load="",
        num_rollout=120,
        save_interval=5,
        checkpoint_dir=str(tmp_path),
    )
    calls = []

    def run(argv):
        calls.append(argv)
        if not initial_failure:
            (tmp_path / "latest_checkpointed_iteration.txt").write_text("4")

    monkeypatch.setattr(launch, "_run", run)
    argv = ["python", "recipe.py", "--load", "", "--num-rollout", "120"]
    if initial_failure:
        with pytest.raises(RuntimeError, match="expected checkpoint"):
            launch._verify_training_resume(args, argv, tmp_path)
        assert len(calls) == 1
    else:
        launch._verify_training_resume(args, argv, tmp_path)
        assert len(calls) == 2
        assert calls[0][-1] == "--debug-exit-after-rollout 5"
        assert calls[1][calls[1].index("--load") + 1] == str(tmp_path)
        assert "--no-skip-eval-before-train" in calls[1]
        assert "--extra-args" not in calls[1]
        assert calls[1][calls[1].index("--num-rollout") + 1] == "120"
