import asyncio
import json
from types import SimpleNamespace

import pytest
import torch

import train_sft
from miles.backends.megatron_utils.sft_validation import _masked_totals
from miles.utils.sft_checkpoint import begin_save, finish_save, recover_checkpoint


def _complete_model_and_cursor(root, iteration):
    path = root / f"iter_{iteration:07d}"
    (path / ".metadata").write_text("test metadata")
    cursor = root / "rollout" / str(iteration) / "data_source"
    cursor.mkdir(parents=True)
    (cursor / "state.pt").write_text("test cursor")
    (root / "latest_checkpointed_iteration.txt").write_text(str(iteration))
    return path


def _save(root, iteration, state, keep_best=True):
    begin_save(root, iteration, state)
    path = _complete_model_and_cursor(root, iteration)
    finish_save(root, iteration, state, keep_best=keep_best)
    return path


def test_retention_keeps_best_latest_and_cursors(tmp_path):
    state = dict(best_loss=0.5, best_iteration=99)
    for iteration in (99, 199, 299):
        _save(tmp_path, iteration, state)
    assert {p.name for p in tmp_path.glob("iter_*")} == {"iter_0000099", "iter_0000299"}
    assert {p.name for p in (tmp_path / "rollout").iterdir()} == {"99", "299"}
    assert recover_checkpoint(tmp_path) == dict(iteration=299, state=state)


@pytest.mark.parametrize("failure", ["before_model", "after_tracker", "after_commit"])
def test_interrupted_save_recovers_consistent_model_cursor_and_state(tmp_path, failure):
    old = dict(best_loss=0.5, best_iteration=99)
    new = dict(best_loss=0.4, best_iteration=199)
    _save(tmp_path, 99, old)
    begin_save(tmp_path, 199, new)
    if failure != "before_model":
        _complete_model_and_cursor(tmp_path, 199)
    if failure == "after_commit":
        finish_save(tmp_path, 199, new, keep_best=True)
        # A crash between manifest replacement and pending marker deletion.
        (tmp_path / "sft_pending.json").write_text(json.dumps(dict(iteration=199)))
    commit = recover_checkpoint(tmp_path)
    expected = 199 if failure == "after_commit" else 99
    assert commit == dict(iteration=expected, state=new if expected == 199 else old)
    assert int((tmp_path / "latest_checkpointed_iteration.txt").read_text()) == expected
    assert (tmp_path / "rollout" / str(expected) / "data_source" / "state.pt").exists()
    assert not (tmp_path / "sft_pending.json").exists()
    if expected == 99:
        assert not (tmp_path / "iter_0000199").exists()


def test_interrupted_first_save_recovers_fresh_start(tmp_path):
    state = dict(best_loss=0.5, best_iteration=0)
    begin_save(tmp_path, 0, state)
    _complete_model_and_cursor(tmp_path, 0)
    assert recover_checkpoint(tmp_path)["iteration"] == -1
    assert not (tmp_path / "latest_checkpointed_iteration.txt").exists()
    assert not (tmp_path / "iter_0000000").exists()


def test_incomplete_save_never_prunes_previous_checkpoint(tmp_path):
    state = dict(best_loss=0.5, best_iteration=99)
    old = _save(tmp_path, 99, state)
    begin_save(tmp_path, 199, state)
    with pytest.raises(RuntimeError, match="Incomplete"):
        finish_save(tmp_path, 199, state, keep_best=True)
    assert old.exists()
    assert json.loads((tmp_path / "sft_latest.json").read_text())["iteration"] == 99


def test_missing_cursor_prevents_commit(tmp_path):
    state = dict(best_loss=0.5, best_iteration=0)
    begin_save(tmp_path, 0, state)
    (tmp_path / "iter_0000000" / ".metadata").touch()
    with pytest.raises(RuntimeError, match="data cursor"):
        finish_save(tmp_path, 0, state, keep_best=True)
    assert not (tmp_path / "sft_latest.json").exists()


def test_retention_leaves_non_sft_checkpoints_untouched(tmp_path):
    unrelated = tmp_path / "iter_0000008"
    unrelated.mkdir()
    _save(tmp_path, 9, dict(best_loss=0.5, best_iteration=9), keep_best=False)
    assert unrelated.exists()


def test_existing_non_sft_tracker_requires_explicit_migration(tmp_path):
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("9")
    with pytest.raises(ValueError, match="without an SFT commit manifest"):
        recover_checkpoint(tmp_path)


@pytest.mark.parametrize("async_render", [False, True])
def test_validation_tail_is_masked_and_rendering_is_batch_bounded(async_render):
    class Dataset:
        samples = [SimpleNamespace(prompt=str(i)) for i in range(5)]

        def __len__(self):
            return len(self.samples)

    calls = []

    def render(args, rollout_id, buffer, evaluation=False):
        calls.append(rollout_id)
        rows = buffer.get_samples(4)
        for (sample,) in rows:
            sample.response_length = 2
            sample.loss_mask = [1, 1]
        return rows

    async def render_async(*args, **kwargs):
        return render(*args, **kwargs)

    async def run():
        batches = train_sft._validation_batches(
            SimpleNamespace(global_batch_size=4), Dataset(), render_async if async_render else render
        )
        assert not calls
        first = await anext(batches)
        assert len(calls) == 1
        last = await anext(batches)
        assert len(calls) == 2
        with pytest.raises(StopAsyncIteration):
            await anext(batches)
        assert sum(sum(s.loss_mask) for s in first + last) == 10
        assert last[0].loss_mask == [1, 1]
        assert all(s.loss_mask == [0, 0] for s in last[1:])
        assert not hasattr(Dataset.samples[0], "loss_mask")

    asyncio.run(run())


def test_validation_uses_training_dataset_configuration_without_mutation(monkeypatch):
    captured = []
    monkeypatch.setattr(
        train_sft,
        "load_function",
        lambda path: lambda args: captured.append(args) or SimpleNamespace(dataset=SimpleNamespace(samples=["val"])),
    )
    args = SimpleNamespace(
        prompt_data="train",
        sft_validation_data="val",
        sft_validation_data_source_path="validation_source",
        rollout_shuffle=True,
        dump_details="dumps",
        label_key="answer",
        tool_key="tools",
        metadata_key="metadata",
    )
    assert train_sft._validation_dataset(args).samples == ["val"]
    assert captured[0].prompt_data == "val"
    assert not captured[0].rollout_shuffle
    assert captured[0].dump_details is None
    assert captured[0].label_key == args.label_key
    assert captured[0].tool_key == args.tool_key
    assert args.prompt_data == "train"
    assert args.rollout_shuffle


def test_validation_reduction_counts_disjoint_dp_cp_tokens_once():
    partitions = [([-1.0, -2.0], [1, 1]), ([-3.0, float("-inf")], [1, 0]), ([-4.0], [1])]
    total = torch.zeros(2, dtype=torch.float64)
    for scores, masks in partitions:
        for tp_rank in range(4):
            total += _masked_totals(
                [torch.tensor(scores)], [torch.tensor(masks)], contributes=tp_rank == 0, device="cpu"
            )
    assert total.tolist() == [10.0, 4.0]


def test_empty_and_fully_masked_validation_shards_have_zero_totals():
    for scores, masks in [(torch.empty(0), torch.empty(0)), (torch.tensor([float("nan")]), torch.tensor([0]))]:
        assert _masked_totals([scores], [masks], contributes=True, device="cpu").tolist() == [0, 0]


def test_new_sft_methods_have_valid_rpc_wire_schemas():
    import ast
    from pathlib import Path

    from miles.ray.train.group import TrainerController
    from miles.utils.object_store import StoreObjectRef
    from miles.utils.workers.rpc.common.metadata import collect_rpc_method_specs

    assert "evaluate_sft" in collect_rpc_method_specs(TrainerController)
    path = Path(train_sft.__file__).parent / "miles/backends/megatron_utils/actor.py"
    tree = ast.parse(path.read_text())
    actor = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MegatronTrainRayActor")
    method = next(n for n in actor.body if isinstance(n, ast.FunctionDef) and n.name == "evaluate_sft")
    namespace = {"StoreObjectRef": StoreObjectRef}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    backend = type("Backend", (), {"evaluate_sft": namespace["evaluate_sft"]})
    spec = collect_rpc_method_specs(backend)["evaluate_sft"]
    payload = spec.serializer.encode_result({"nll_sum": 10.0, "token_count": 4.0})
    assert spec.serializer.decode_result(payload) == {"nll_sum": 10.0, "token_count": 4.0}


def _args(**overrides):
    values = dict(
        train_backend="megatron",
        ckpt_format="torch_dist",
        indep_dp=False,
        megatron_config=None,
        debug_train_only=True,
        loss_type="sft_loss",
        async_save=False,
        offload_train=False,
        eval_num_gpus=0,
        use_critic=False,
        fully_async=False,
        global_batch_size=4,
        rollout_batch_size=4,
        n_samples_per_prompt=1,
        save="/tmp/sft",
        requested_load="/tmp/sft",
        save_retain_interval=None,
        sft_eval_interval=2,
        save_interval=2,
        no_save_optim=False,
        no_save_rng=False,
        start_rollout_id=None,
        ckpt_step=None,
        rollout_global_dataset=True,
        data_source_path="miles.rollout.data_source.RolloutDataSource",
        sft_validation_data_source_path=None,
        use_dynamic_global_batch_size=False,
        custom_convert_samples_to_train_data_path=None,
        custom_reward_post_process_path=None,
        qkv_format="thd",
        allgather_cp=False,
        rollout_function_path="miles.rollout.sft_rollout.generate_rollout",
    )
    return SimpleNamespace(**(values | overrides))


@pytest.mark.parametrize(
    "override",
    [
        dict(indep_dp=True),
        dict(train_backend="fsdp"),
        dict(ckpt_format="torch"),
        dict(offload_train=True),
        dict(async_save=True),
        dict(eval_num_gpus=8),
        dict(global_batch_size=8),
        dict(requested_load=None),
        dict(no_save_optim=True),
        dict(no_save_rng=True),
        dict(start_rollout_id=42),
        dict(rollout_global_dataset=False),
        dict(use_dynamic_global_batch_size=True),
        dict(allgather_cp=True),
        dict(custom_convert_samples_to_train_data_path="custom"),
        dict(megatron_config="multiple.yaml"),
    ],
)
def test_unsupported_configurations_fail_before_worker_allocation(override):
    with pytest.raises(ValueError):
        train_sft._validate_configuration(_args(**override))


def test_configuration_accepts_fresh_initialization_resolved_by_megatron():
    args = _args(start_rollout_id=0)
    args.load = "initial-reference-weights"
    args.no_load_optim = True
    args.no_load_rng = True
    args.finetune = True
    assert callable(train_sft._validate_configuration(args))


def test_validation_restores_rng_nested_modes_and_timers_on_failure(monkeypatch):
    import sys
    from contextlib import nullcontext

    from miles.backends.megatron_utils.sft_validation import evaluate_sft

    model = torch.nn.Sequential(torch.nn.Dropout(), torch.nn.Dropout())
    model[1].eval()
    before_modes = [module.training for module in model.modules()]
    timer = object()
    config = SimpleNamespace(timers=timer)
    restored = []
    heartbeat = []
    random_state = SimpleNamespace(restore=lambda: restored.append(True))

    def forward(*args, **kwargs):
        model.eval()
        config.timers = None
        raise RuntimeError("simulated pipeline failure")

    mocks = {
        "megatron.core.utils": SimpleNamespace(get_model_config=lambda model: config),
        "miles.backends.megatron_utils.initialize": SimpleNamespace(
            RandomState=SimpleNamespace(capture=lambda: random_state)
        ),
        "miles.backends.megatron_utils.model": SimpleNamespace(forward_only=forward),
        "miles.backends.training_utils.data.rollout": SimpleNamespace(
            get_rollout_data=lambda *args: ({}, nullcontext()), get_data_iterator=lambda *args: ([], [])
        ),
    }
    for name, module in mocks.items():
        monkeypatch.setitem(sys.modules, name, module)
    actor = SimpleNamespace(
        args=SimpleNamespace(loss_type="sft_loss", offload_train=False),
        model=[model],
        _heartbeat=SimpleNamespace(bump=lambda: heartbeat.append(True)),
    )
    with pytest.raises(RuntimeError, match="pipeline failure"):
        evaluate_sft(actor, 0, None)
    assert [module.training for module in model.modules()] == before_modes
    assert config.timers is timer
    assert restored == [True]
    assert heartbeat == [True, True]


def test_custom_training_source_requires_explicit_validation_source():
    with pytest.raises(ValueError, match="explicit --sft-validation-data-source-path"):
        train_sft._validate_configuration(_args(data_source_path="custom.EpochDataSource"))
    assert callable(
        train_sft._validate_configuration(
            _args(
                data_source_path="custom.EpochDataSource",
                sft_validation_data_source_path="miles.rollout.data_source.RolloutDataSource",
            )
        )
    )


def test_driver_fresh_save_resume_and_missing_tracker_recovery(tmp_path, monkeypatch):
    from miles.backends.megatron_utils.megatron_config import resolve_args_checkpoint_load

    trained = []
    validated = []

    class Actor:
        async def get_train_parallel_config(self):
            return {}

        async def train(self, iteration, pack):
            trained.append(iteration)
            return None

        async def save_model(self, iteration, force_sync):
            assert force_sync
            (tmp_path / f"iter_{iteration:07d}" / ".metadata").touch()
            (tmp_path / "latest_checkpointed_iteration.txt").write_text(str(iteration))

    class Executor:
        async def get_num_rollout_per_epoch(self):
            return 2

        async def get(self, iteration):
            return None

        async def save(self, iteration):
            cursor = tmp_path / "rollout" / str(iteration) / "data_source"
            cursor.mkdir(parents=True)
            (cursor / "state.pt").write_text(str(iteration))

    async def create_rollout(args):
        return None, Executor(), None

    async def create_train(args, executor):
        tracker = tmp_path / "latest_checkpointed_iteration.txt"
        if tracker.exists():
            assert args.load == str(tmp_path)
            assert not args.no_load_optim and not args.no_load_rng and not args.finetune
            args.start_rollout_id = int(tracker.read_text()) + 1
        else:
            assert args.load == "hf-initializer"
            args.start_rollout_id = 0
        return Actor(), None

    async def validate(args, actor, parallel, dataset, render, step):
        validated.append(step)
        return 1.0 / (step + 1)

    monkeypatch.setattr(train_sft, "create_rollout_components", create_rollout)
    monkeypatch.setattr(train_sft, "create_training_models", create_train)
    monkeypatch.setattr(train_sft, "_validation_dataset", lambda args: [1])
    monkeypatch.setattr(train_sft, "_validate", validate)
    monkeypatch.setattr(train_sft, "init_orchestration_script", lambda *a, **kw: None)
    monkeypatch.setattr(train_sft, "remove_train_output_refs", lambda *a: None)
    monkeypatch.setattr(train_sft, "remove_rollout_data_refs", lambda *a: None)
    monkeypatch.setattr(train_sft.tracking, "log", lambda *a, **kw: None)

    def args_for(end):
        args = _args(
            save=str(tmp_path),
            requested_load=str(tmp_path),
            num_rollout=end,
            sft_keep_best=True,
            load=str(tmp_path),
            no_load_optim=False,
            no_load_rng=False,
            finetune=False,
            megatron_to_hf_mode="bridge",
            hf_checkpoint="hf-initializer",
            ref_load=None,
            debug_exit_after_rollout=None,
        )
        resolve_args_checkpoint_load(args)  # Exactly the transformation parse_args applies.
        return args

    disposer = SimpleNamespace(add=lambda *a: None)
    asyncio.run(train_sft._train(args_for(2), disposer=disposer))
    assert trained == [0, 1]
    assert validated == [0, 2]
    assert recover_checkpoint(tmp_path)["iteration"] == 1
    asyncio.run(train_sft._train(args_for(4), disposer=disposer))
    assert trained == [0, 1, 2, 3]
    assert validated == [0, 2, 4]
    (tmp_path / "latest_checkpointed_iteration.txt").unlink()
    asyncio.run(train_sft._train(args_for(6), disposer=disposer))
    assert trained == list(range(6))
    assert recover_checkpoint(tmp_path)["iteration"] == 5
    bad = args_for(8)
    bad.no_load_optim = True
    with pytest.raises(ValueError, match="restore optimizer/RNG"):
        asyncio.run(train_sft._train(bad, disposer=disposer))
    assert trained == list(range(6))


def test_custom_data_source_checkpoint_format_is_not_assumed(tmp_path):
    state = dict(best_loss=0.5, best_iteration=0)
    begin_save(tmp_path, 0, state)
    _complete_model_and_cursor(tmp_path, 0)
    cursor = tmp_path / "rollout" / "0" / "data_source"
    (cursor / "state.pt").rename(cursor / "cursor.json")
    finish_save(tmp_path, 0, state, keep_best=True)
    assert recover_checkpoint(tmp_path)["iteration"] == 0
