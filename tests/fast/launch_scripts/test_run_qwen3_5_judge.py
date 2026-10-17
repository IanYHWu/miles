import shlex

import pytest

from tests.fast.launch_scripts.py_harness import freeze_environment, import_launch_script, install_command_recorder
from tests.fast.launch_scripts.sh_harness import REPO_ROOT


@pytest.mark.parametrize("smoke,expected_gpus,expected_updates", [(False, "8", "100"), (True, "4", "2")])
def test_judged_recipe_uses_async_cispo_without_logging_credentials(
    monkeypatch, smoke, expected_gpus, expected_updates
):
    freeze_environment(monkeypatch)
    monkeypatch.setenv("MILES_SCRIPT_EXTERNAL_RAY", "1")
    monkeypatch.setenv("WANDB_API_KEY", "secret-must-not-appear-in-command")
    recording = install_command_recorder(monkeypatch)
    launcher = import_launch_script(REPO_ROOT / "scripts/run_qwen3_5_judge.py")
    args = launcher.ScriptArgs(smoke=smoke, private_pythonpath="/private/helpers")
    flags = shlex.split(launcher._train_args(args))
    assert flags[flags.index("--actor-num-gpus-per-node") + 1] == expected_gpus
    assert flags[flags.index("--num-rollout") + 1] == expected_updates
    assert flags[flags.index("--policy-loss-type") + 1] == "cispo"
    assert "--fully-async" in flags and "--calculate-per-token-loss" in flags
    assert "--max-weight-staleness" not in flags
    launcher.execute(args)
    commands = "\n".join(recording.commands)
    assert "train_async.py" in commands
    assert "secret-must-not-appear-in-command" not in commands
    assert "--wandb-key" not in commands
    assert "/private/helpers" in commands


def test_122b_smoke_uses_full_model_and_routing_replay():
    launcher = import_launch_script(REPO_ROOT / "scripts/run_qwen3_5_judge.py")
    args = launcher.ScriptArgs(smoke=True, model_name="Qwen3.5-122B-A10B")
    flags = shlex.split(launcher._train_args(args))
    assert "--colocate" in flags and "--rollout-num-gpus" not in flags
    assert "--use-rollout-routing-replay" in flags
    assert flags[flags.index("--pause-generation-mode") + 1] == "retract"
    assert args.actor_gpus == args.rollout_tp == 8
    assert args.update_interval == 1


def test_async_smoke_keeps_rollout_and_training_on_separate_gpus():
    launcher = import_launch_script(REPO_ROOT / "scripts/run_qwen3_5_judge.py")
    args = launcher.ScriptArgs(async_smoke=True)
    flags = shlex.split(launcher._train_args(args))
    assert args.smoke and args.num_rollout > 2 * args.update_interval
    assert "--colocate" not in flags
    assert args.actor_gpus + args.rollout_gpus == 8
    assert flags[flags.index("--pause-generation-mode") + 1] == "in_place"
    assert args.concurrency > args.rollout_batch_size * args.samples_per_prompt
    with pytest.raises(ValueError, match="separate 9B"):
        launcher.ScriptArgs(async_smoke=True, model_name="Qwen3.5-122B-A10B")


def test_production_retains_only_latest_checkpoint_and_accepts_three_rollout_nodes():
    launcher = import_launch_script(REPO_ROOT / "scripts/run_qwen3_5_judge.py")
    args = launcher.ScriptArgs(num_rollout=120, rollout_gpus=24)
    flags = shlex.split(launcher._train_args(args))
    retain = int(flags[flags.index("--save-retain-interval") + 1])
    assert retain > args.num_rollout and retain % args.save_interval == 0
    assert flags[flags.index("--rollout-num-gpus") + 1] == "24"
    assert "--async-save" not in flags and "--no-save-optim" not in flags
    assert "--max-weight-staleness" not in flags


def test_resume_can_save_into_original_checkpoint_root():
    launcher = import_launch_script(REPO_ROOT / "scripts/run_qwen3_5_judge.py")
    args = launcher.ScriptArgs(
        num_rollout=120,
        load="/existing/checkpoints",
        checkpoint_dir="/existing/checkpoints",
        skip_eval_before_train=False,
    )
    flags = shlex.split(launcher._train_args(args))
    assert flags[flags.index("--load") + 1] == flags[flags.index("--save") + 1] == "/existing/checkpoints"
    assert flags[flags.index("--num-rollout") + 1] == "120"
    assert "--skip-eval-before-train" not in flags
    assert flags[flags.index("--save-interval") + 1] == "5"
    assert flags[flags.index("--save-retain-interval") + 1] == "125"
