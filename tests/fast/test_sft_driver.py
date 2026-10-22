from types import SimpleNamespace

import pytest

import train_sft


def _checkpoint(root, iteration):
    path = root / f"iter_{iteration:07d}"
    path.mkdir()
    (path / ".metadata").write_text("test metadata")
    (root / "rollout" / str(iteration)).mkdir(parents=True)
    (root / "latest_checkpointed_iteration.txt").write_text(str(iteration))
    return path


def test_retention_keeps_best_and_latest_with_data_cursors(tmp_path):
    for iteration in (99, 199, 299):
        _checkpoint(tmp_path, iteration)
    state = dict(best_loss=0.5, best_iteration=99)
    train_sft._finish_save(tmp_path, 299, state, True)
    assert {p.name for p in tmp_path.glob("iter_*")} == {"iter_0000099", "iter_0000299"}
    assert {p.name for p in (tmp_path / "rollout").iterdir()} == {"99", "299"}
    assert train_sft._read_state(tmp_path, 299) == state


def test_failed_save_cannot_prune_existing_weights(tmp_path):
    old = _checkpoint(tmp_path, 99)
    new = _checkpoint(tmp_path, 199)
    (new / ".metadata").unlink()
    with pytest.raises(RuntimeError, match="Incomplete"):
        train_sft._finish_save(tmp_path, 199, dict(best_loss=0.5, best_iteration=199), True)
    assert old.exists()


def test_validation_tail_padding_is_masked_without_dropping_real_rows(monkeypatch):
    class Dataset:
        def __init__(self, *a, **kw):
            self.samples = [SimpleNamespace(prompt=str(i)) for i in range(5)]

        def __len__(self):
            return len(self.samples)

    def render(args, rollout_id, buffer):
        groups = buffer.get_samples(4)
        for (sample,) in groups:
            sample.response_length = 2
            sample.loss_mask = [1, 1]
        return groups

    monkeypatch.setattr(train_sft, "Dataset", Dataset)
    monkeypatch.setattr(train_sft, "load_tokenizer", lambda *a, **kw: None)
    monkeypatch.setattr(train_sft, "load_function", lambda *a: render)
    args = SimpleNamespace(
        hf_checkpoint="model",
        sft_validation_data="val",
        input_key="messages",
        rollout_function_path="render",
        global_batch_size=4,
    )
    batches = list(train_sft._validation_batches(args))
    assert len(batches) == 2
    assert sum(sum(s.loss_mask) for b in batches for s in b) == 10
    assert batches[1][0].loss_mask == [1, 1]
    assert all(s.loss_mask == [0, 0] for s in batches[1][1:])


def test_validation_reduction_counts_disjoint_dp_cp_tokens_once():
    import torch
    from miles.backends.megatron_utils.sft_validation import _masked_totals

    partitions = [([-1.0, -2.0], [1, 1]), ([-3.0, -9.0], [1, 0]), ([-4.0], [1])]
    total = torch.zeros(2, dtype=torch.float64)
    for scores, masks in partitions:
        for tp_rank in range(4):
            total += _masked_totals(
                [torch.tensor(scores)], [torch.tensor(masks)], contributes=tp_rank == 0, device="cpu"
            )
    assert total.tolist() == [10.0, 4.0]
    assert total[0] / total[1] == 2.5


def test_new_sft_methods_have_valid_rpc_wire_schemas():
    import ast
    from pathlib import Path
    from miles.ray.train.group import TrainerController
    from miles.utils.object_store import StoreObjectRef
    from miles.utils.workers.rpc.common.metadata import collect_rpc_method_specs

    assert "evaluate_sft" in collect_rpc_method_specs(TrainerController)
    # Load the real backend method without importing Megatron/CUDA on the CPU runner.
    path = Path(train_sft.__file__).parent / "miles/backends/megatron_utils/actor.py"
    tree = ast.parse(path.read_text())
    actor = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MegatronTrainRayActor")
    method = next(n for n in actor.body if isinstance(n, ast.FunctionDef) and n.name == "evaluate_sft")
    namespace = {"StoreObjectRef": StoreObjectRef}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    backend = type("Backend", (), {"evaluate_sft": namespace["evaluate_sft"]})
    spec = collect_rpc_method_specs(backend)["evaluate_sft"]
    payload = spec.serializer.encode_result({"nll_sum": 10., "token_count": 4.})
    assert spec.serializer.decode_result(payload) == {"nll_sum": 10., "token_count": 4.}
