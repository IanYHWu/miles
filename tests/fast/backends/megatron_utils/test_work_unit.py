"""A work unit's backward runs under the gradient reduction its actor gives, on the config Megatron's schedule reads;
a forward-only unit leaves Megatron's own in place."""

from argparse import Namespace
from types import SimpleNamespace

import pytest

from miles.backends.megatron_utils import work_unit


class _Iterator:
    def reset(self) -> None:
        pass


class _Chunk:
    def train(self) -> None:
        pass


@pytest.fixture
def reductions_seen(monkeypatch):
    """The finalize_model_grads_func each pass ran under; the setup restores Megatron's every call, as the real one."""
    config = SimpleNamespace()
    seen = []

    def setup_train_iteration_config(*_args, **_kwargs):
        config.finalize_model_grads_func = "megatron"
        return config

    def run_forward_backward_pass(*_args, **_kwargs):
        seen.append(config.finalize_model_grads_func)
        return [{"per_datum": []}]

    monkeypatch.setattr(work_unit, "get_data_iterator", lambda *_args: ([_Iterator()], [1]))
    monkeypatch.setattr(work_unit, "setup_train_iteration_config", setup_train_iteration_config)
    monkeypatch.setattr(work_unit, "run_forward_backward_pass", run_forward_backward_pass)
    monkeypatch.setattr(work_unit, "reset_grad_metadata_keep_grads", lambda _model: None)
    monkeypatch.setattr(
        work_unit, "DumperMegatronUtil", lambda *_args, **_kwargs: SimpleNamespace(finalize=lambda _model: None)
    )
    monkeypatch.setattr(work_unit, "get_parallel_state", lambda: SimpleNamespace(is_pp_last_stage=False))
    return seen


def test_each_backward_runs_under_the_given_reduction_around_a_forward_only_unit(reductions_seen):
    def deferred(*_args):
        pass

    for forward_only in (False, True, False):
        work_unit.run_forward_backward(
            Namespace(),
            1,
            [_Chunk()],
            {},
            forward_only=forward_only,
            finalize_model_grads_func=None if forward_only else deferred,
        )

    assert reductions_seen == [deferred, "megatron", deferred]
