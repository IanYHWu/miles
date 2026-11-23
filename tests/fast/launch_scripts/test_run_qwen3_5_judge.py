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


def test_122b_dedicated_training_preserves_full_checkpoints_and_async_replay():
    launcher = import_launch_script(REPO_ROOT / "scripts/run_qwen3_5_judge.py")
    args = launcher.ScriptArgs(
        model_name="Qwen3.5-122B-A10B",
        actor_num_nodes=4,
        rollout_gpus=32,
        train_tp=4,
        train_cp=2,
        train_ep=32,
        rollout_tp=8,
        optimizer_cpu_offload=False,
        num_rollout=120,
        checkpoint_dir="/groups/project/checkpoints/run",
    )
    flags = shlex.split(launcher._train_args(args))
    for flag, expected in {
        "--actor-num-nodes": "4",
        "--actor-num-gpus-per-node": "8",
        "--tensor-model-parallel-size": "4",
        "--context-parallel-size": "2",
        "--expert-model-parallel-size": "32",
        "--expert-tensor-parallel-size": "1",
        "--rollout-num-gpus": "32",
        "--rollout-num-gpus-per-engine": "8",
        "--pause-generation-mode": "in_place",
        "--save-retain-interval": "125",
        "--save": "/groups/project/checkpoints/run",
    }.items():
        assert flags[flags.index(flag) + 1] == expected
    assert "--use-rollout-routing-replay" in flags
    assert "--namespaced-radix-cache" in flags
    for flag in (
        "--colocate",
        "--no-save-optim",
        "--optimizer-cpu-offload",
        "--streaming-model-checkpoint-save",
        "--offload-train-target",
    ):
        assert flag not in flags


def test_invalid_122b_parallelism_fails_before_starting_workers():
    launcher = import_launch_script(REPO_ROOT / "scripts/run_qwen3_5_judge.py")
    with pytest.raises(ValueError, match="explicit expert-parallel"):
        launcher.ScriptArgs(model_name="Qwen3.5-122B-A10B")
    with pytest.raises(ValueError, match="divisible"):
        launcher.ScriptArgs(actor_num_nodes=4, train_ep=24)


def test_the_recipe_wires_miles_judge_reward_and_its_score_key(tmp_path, monkeypatch):
    """The recipe hands --llm-judge-config to miles.rollout.llm_judge.reward and trains on its "score"."""
    import asyncio
    import json
    import sys
    from types import ModuleType, SimpleNamespace

    from miles.rollout import llm_judge
    from miles.utils.function_registry import load_function

    launcher = import_launch_script(REPO_ROOT / "scripts/run_qwen3_5_judge.py")
    flags = shlex.split(launcher._train_args(launcher.ScriptArgs(judge_config=str(tmp_path / "judge.json"))))
    assert load_function(flags[flags.index("--custom-rm-path") + 1]) is llm_judge.reward
    assert flags[flags.index("--llm-judge-config") + 1] == str(tmp_path / "judge.json")
    reward_key = flags[flags.index("--reward-key") + 1]

    class Adapter:
        def __init__(self, config):
            self.config = config

        def prepare(self, sample):  # an answer the application rejects: no HTTP call
            return {reward_key: 0.0, "fault": self.config["fault"]}

        def parse_reply(self, content, context):
            raise AssertionError("not reached")

    module = ModuleType("abci_test_adapter")
    module.make = Adapter
    monkeypatch.setitem(sys.modules, "abci_test_adapter", module)
    (tmp_path / "judge.json").write_text(
        json.dumps(
            {
                "endpoint_url": "https://judge.invalid/v1/chat/completions",
                "model": "judge",
                "adapter_factory": "abci_test_adapter.make",
                "adapter_config": {"fault": "no answer"},
            }
        )
    )
    args = SimpleNamespace(llm_judge_config=str(tmp_path / "judge.json"))
    reward = asyncio.run(llm_judge.reward(args, SimpleNamespace(response="")))
    assert reward == {"score": 0.0, "fault": "no answer"}


def test_the_smoke_reward_fixture_answers_the_score_key():
    import asyncio
    from types import SimpleNamespace

    from tests.manual.judged_smoke_fixture import reward

    assert [asyncio.run(reward(None, SimpleNamespace(index=i))) for i in range(2)] == [{"score": 0.0}, {"score": 1.0}]


def test_dp_attention_recipe_preserves_rl_settings_and_uses_one_engine_per_node():
    launcher = import_launch_script(REPO_ROOT / "scripts/run_qwen3_5_judge.py")
    args = launcher.ScriptArgs(
        model_name="Qwen3.5-122B-A10B",
        actor_num_nodes=4,
        rollout_gpus=32,
        train_cp=2,
        train_ep=32,
        rollout_tp=8,
        rollout_dp=4,
        rollout_ep=8,
        rollout_max_running_requests=192,
        rollout_max_mamba_cache_size=1280,
        rollout_router_policy="round_robin",
        check_initial_weights=True,
        optimizer_cpu_offload=False,
        num_rollout=120,
    )
    flags = shlex.split(launcher._train_args(args))
    for flag, value in {
        "--rollout-num-gpus-per-engine": "8",
        "--sglang-data-parallel-size": "4",
        "--sglang-ep-size": "8",
        "--sglang-max-running-requests": "192",
        "--sglang-max-mamba-cache-size": "1280",
        "--sglang-router-policy": "round_robin",
        "--rollout-max-response-len": "92160",
        "--async-max-concurrent-samples": "1024",
        "--update-weights-interval": "2",
        "--pause-generation-mode": "in_place",
        "--global-batch-size": "512",
    }.items():
        assert flags.count(flag) == 1
        assert flags[flags.index(flag) + 1] == value
    assert "--sglang-enable-dp-attention" in flags
    assert "--check-weight-update-equal" in flags
    assert "--use-rollout-routing-replay" in flags


def test_invalid_rollout_topology_and_resume_comparison_are_rejected():
    launcher = import_launch_script(REPO_ROOT / "scripts/run_qwen3_5_judge.py")
    for overrides in ({"rollout_dp": 3}, {"rollout_ep": 3}):
        with pytest.raises(ValueError, match="divisible"):
            launcher.ScriptArgs(rollout_tp=8, **overrides)
    with pytest.raises(ValueError, match="fresh run"):
        launcher.ScriptArgs(check_initial_weights=True, load="/checkpoint")


@pytest.mark.parametrize("load", ["", "/existing/checkpoints"])
def test_constant_rate_continuation_restores_scheduler_without_resetting_optimizer(load):
    launcher = import_launch_script(REPO_ROOT / "scripts/run_qwen3_5_judge.py")
    args = launcher.ScriptArgs(num_rollout=240, load=load)
    flags = shlex.split(launcher._train_args(args))
    assert ("--use-checkpoint-opt-param-scheduler" in flags) == bool(load)
    assert flags[flags.index("--num-rollout") + 1] == "240"
    assert flags[flags.index("--lr-decay-style") + 1] == "constant"
    assert flags[flags.index("--save-retain-interval") + 1] == "245"
    for flag in ("--override-opt-param-scheduler", "--no-load-optim", "--finetune"):
        assert flag not in flags
