import shlex

import pytest

from tests.fast.launch_scripts.py_harness import (
    freeze_environment,
    import_launch_script,
    install_command_recorder,
)
from tests.fast.launch_scripts.sh_harness import REPO_ROOT


def test_sft_uses_all_96_gpus_without_inference_or_rl(monkeypatch):
    freeze_environment(monkeypatch)
    monkeypatch.setenv("MILES_SCRIPT_EXTERNAL_RAY", "1")
    monkeypatch.setenv("WANDB_API_KEY", "do-not-log-this")
    recording = install_command_recorder(monkeypatch)
    module = import_launch_script(REPO_ROOT / "scripts/run_qwen3_5_sft.py")
    args = module.ScriptArgs(private_pythonpath="/private/code")
    flags = shlex.split(module._train_args(args))
    for flag, value in {
        "--actor-num-nodes": "12",
        "--tensor-model-parallel-size": "4",
        "--context-parallel-size": "2",
        "--pipeline-model-parallel-size": "3",
        "--expert-model-parallel-size": "32",
        "--global-batch-size": "128",
        "--num-rollout": "4128",
        "--lr": "2e-5",
        "--min-lr": "2e-7",
    }.items():
        assert flags[flags.index(flag) + 1] == value
    for flag in (
        "--enable-mtp-training",
        "--use-rollout-routing-replay",
        "--fully-async",
        "--optimizer-cpu-offload",
    ):
        assert flag not in flags
    module.execute(args)
    commands = "\n".join(recording.commands)
    assert "train_sft.py" in commands and "/private/code" in commands
    assert "do-not-log-this" not in commands and "--wandb-key" not in commands


def test_sft_rejects_12_way_dp_with_batch_128():
    module = import_launch_script(REPO_ROOT / "scripts/run_qwen3_5_sft.py")
    with pytest.raises(ValueError, match="Global batch"):
        module.ScriptArgs(train_pp=1)


def test_sft_bootstrap_rejects_a_preserved_rl_checkpoint(tmp_path):
    from types import SimpleNamespace
    from tools.abci.launch_sft import _preflight

    model = tmp_path / "model"
    model.mkdir()
    for name in ("config.json", "tokenizer_config.json"):
        (model / name).write_text("{}")
    validation = tmp_path / "validation.parquet"
    validation.touch()
    checkpoint = tmp_path / "old-rl"
    checkpoint.mkdir()
    (checkpoint / "latest_checkpointed_iteration.txt").write_text("209")
    with pytest.raises(ValueError, match="non-SFT"):
        _preflight(
            SimpleNamespace(
                model_dir=str(tmp_path),
                model_name="model",
                validation_data=str(validation),
                checkpoint_dir=str(checkpoint),
            )
        )
