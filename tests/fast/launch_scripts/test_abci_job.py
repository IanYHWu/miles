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
    argv = ["python", "recipe.py", "--load", "", "--num-rollout", "120", "--check-initial-weights"]
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
        assert "--check-initial-weights" in calls[0]
        assert "--check-initial-weights" not in calls[1]
        assert calls[1][calls[1].index("--num-rollout") + 1] == "120"


def test_train_forwards_dp_attention_capacity_and_initial_verification(monkeypatch, tmp_path):
    options = {
        "mode": "train",
        "image": "unused",
        "repo": "unused",
        "scratch": str(tmp_path),
        "model-dir": "unused",
        "data-dir": "unused",
        "judge-config": "unused",
        "rollout-tp": "8",
        "rollout-dp": "4",
        "rollout-ep": "8",
        "rollout-max-running-requests": "192",
        "rollout-max-mamba-cache-size": "1280",
        "rollout-router-policy": "round_robin",
    }
    argv = [part for key, value in options.items() for part in ("--" + key, value)]
    args = launch._parser().parse_args(argv + ["--check-initial-weights"])
    monkeypatch.setenv("PBS_JOBID", "123.pbs")
    calls = []
    monkeypatch.setattr(launch, "_run", calls.append)
    launch._execute_recipe(args)
    (recipe,) = calls
    for name in options:
        if name.startswith("rollout-"):
            assert recipe[recipe.index("--" + name) + 1] == options[name]
    assert "--check-initial-weights" in recipe


def _trainer_args(tmp_path, *extra, entrypoint=None, run_config=None):
    staged = tmp_path / "staged"
    staged.mkdir(exist_ok=True)
    for name in ("launcher.py", "run.json"):
        (staged / name).write_text("")
    argv = [
        "--recipe", "trainer", "--image", str(tmp_path), "--repo", str(tmp_path), "--scratch", str(tmp_path / "scratch"),
        "--entrypoint", str(entrypoint or staged / "launcher.py"), "--run-config", str(run_config or staged / "run.json"),
    ]  # fmt: skip
    return launch._parser().parse_args(argv + list(extra))


def test_the_trainer_recipe_runs_its_entrypoint_without_the_judge_preflight(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(launch, "_run", calls.append)
    args = _trainer_args(tmp_path, "--mode", "train")
    launch._preflight(args)
    assert calls == []  # no judge preflight, and no judge inputs needed
    launch._execute_recipe(args)
    assert calls == [[launch.sys.executable, args.entrypoint, "--run-config", args.run_config]]


@pytest.mark.parametrize("missing", ["entrypoint", "run_config"])
def test_the_trainer_recipe_needs_absolute_staged_files(tmp_path, missing):
    args = _trainer_args(tmp_path, **{missing: tmp_path / "absent"})
    with pytest.raises(SystemExit, match="absolute, staged --" + missing.replace("_", "-")):
        launch._preflight(args)
    relative = _trainer_args(tmp_path, **{missing: "staged/relative.py"})
    with pytest.raises(SystemExit, match="absolute"):
        launch._preflight(relative)


def _host_command(monkeypatch, tmp_path, args, nodes):
    nodefile = tmp_path / "nodes"
    nodefile.write_text("".join(f"node{i}\n" for i in range(nodes)))
    monkeypatch.setenv("PBS_JOBID", "789.pbs")
    monkeypatch.setenv("PBS_NODEFILE", str(nodefile))
    monkeypatch.setattr(launch.socket, "gethostbyname", lambda _: "10.0.0.1")
    monkeypatch.setattr(launch.subprocess, "Popen", lambda command: Mock())
    head = Mock()
    monkeypatch.setattr(launch, "_run_head", head)
    launch._host(args)
    return head.call_args.args[0]


def test_the_trainer_recipe_allocates_actor_nodes_with_a_private_home(monkeypatch, tmp_path):
    args = _trainer_args(tmp_path, "--mode", "train", "--actor-nodes", "2", "--rollout-nodes", "0")
    command = _host_command(monkeypatch, tmp_path, args, nodes=2)
    assert "--no-home" not in command
    assert command[command.index("--home") + 1] == f"{tmp_path}/scratch/home:/tmp/miles-home"
    # staged outside /home, /groups and scratch (pytest's tmp_path): bound read-only
    staged = tmp_path / "staged"
    if not staged.is_relative_to("/home"):
        assert f"{staged}:{staged}:ro" in command
    with pytest.raises(SystemExit, match="requires 2 node"):
        _host_command(monkeypatch, tmp_path, args, nodes=3)


def test_the_judge_recipe_still_requires_its_inputs(monkeypatch, tmp_path):
    args = launch._parser().parse_args(["--image", str(tmp_path), "--repo", str(tmp_path), "--scratch", str(tmp_path)])
    with pytest.raises(SystemExit, match="--model-dir, --data-dir, --judge-config"):
        _host_command(monkeypatch, tmp_path, args, nodes=1)


def test_kernel_caches_are_node_local_and_other_caches_stay_in_scratch(monkeypatch, tmp_path):
    args = _trainer_args(tmp_path)  # preflight mode: no Ray
    environ = {"PBS_JOBID": "789.pbs"}
    monkeypatch.setattr(launch.os, "environ", environ)
    monkeypatch.chdir(tmp_path)
    launch._container_role(args)
    for name, leaf in (
        ("TRITON_CACHE_DIR", "triton"),
        ("TORCHINDUCTOR_CACHE_DIR", "inductor"),
        ("CUDA_CACHE_PATH", "cuda"),
    ):
        assert environ[name] == f"/tmp/miles-789.pbs/{leaf}"
    assert environ["HF_HOME"] == f"{tmp_path}/scratch/hf"
