"""VllmWeightUpdater: rank 0 owns the pusher, every rank drains the collective stream, and failures come
back as results instead of killing the trainer cell."""

from unittest.mock import MagicMock, patch

import pytest
import torch

from miles.backends.training_utils.weight_update.vllm_nccl import updater as updater_module
from miles.backends.training_utils.weight_update.vllm_nccl.updater import VllmWeightUpdater

TIMEOUTS = dict(http_timeout=1.0, connect_timeout=2.0, push_timeout=3.0)


@pytest.fixture
def rank(monkeypatch):
    state = {"rank": 0, "barriers": 0}
    monkeypatch.setattr(updater_module.dist, "get_rank", lambda: state["rank"])

    def barrier(group=None):
        state["barriers"] += 1

    monkeypatch.setattr(updater_module.dist, "barrier", barrier)
    monkeypatch.setattr(updater_module, "get_gloo_group", lambda: None)
    monkeypatch.setattr(updater_module.torch.cuda, "synchronize", lambda: None)
    return state


def _stream(consumed):
    def iterate():
        for i in range(3):
            consumed.append(i)
            yield [(f"w{i}", torch.ones(1))]

    return iterate


def test_other_ranks_neither_connect_nor_send_but_drain_the_stream(rank):
    rank["rank"] = 3
    consumed = []
    updater = VllmWeightUpdater(_stream(consumed))
    with patch.object(updater_module, "VllmPusher") as pusher_cls:
        assert updater.connect([{"url": "http://a:1", "world_size": 8}], "nccl.so", **TIMEOUTS) is None
        assert updater.push("1") is None
    pusher_cls.assert_not_called()
    assert consumed == [0, 1, 2] and rank["barriers"] == 1


def test_rank0_connects_and_replaces_an_earlier_connection(rank):
    first, second = MagicMock(), MagicMock()
    first.connect.return_value = second.connect.return_value = {"servers": 1}
    updater = VllmWeightUpdater(_stream([]))
    with patch.object(updater_module, "VllmPusher", side_effect=[first, second]) as pusher_cls:
        assert updater.connect([], "nccl.so", **TIMEOUTS) == {"servers": 1, "ok": True}
        updater.connect([], "nccl.so", **TIMEOUTS, extra_ranks=1, test_batch_delay=0.5)
    first.close.assert_called_once()
    assert pusher_cls.call_args.kwargs == dict(TIMEOUTS, extra_ranks=1, test_batch_delay=0.5)


def test_a_failed_connect_is_reported_with_its_wedged_servers(rank):
    pusher = MagicMock()
    error = TimeoutError("no server connected")
    error.wedged = ["http://a:1"]
    pusher.connect.side_effect = error
    updater = VllmWeightUpdater(_stream([]))
    with patch.object(updater_module, "VllmPusher", return_value=pusher):
        out = updater.connect([], "nccl.so", **TIMEOUTS)
    assert out["ok"] is False and out["wedged"] == ["http://a:1"] and "no server connected" in out["error"]
    pusher.close.assert_called_once()


def test_push_without_a_connection_is_an_error_after_draining(rank):
    consumed = []
    out = VllmWeightUpdater(_stream(consumed)).push("2")
    assert out["ok"] is False and "not connected" in out["error"]
    assert consumed == [0, 1, 2] and rank["barriers"] == 1


def test_a_push_failing_midway_still_finishes_the_collective_stream(rank):
    consumed = []

    def push(stream, version):
        next(stream)
        raise RuntimeError("every server failed during the push")

    pusher = MagicMock()
    pusher.connect.return_value = {}
    pusher.push.side_effect = push
    updater = VllmWeightUpdater(_stream(consumed))
    with patch.object(updater_module, "VllmPusher", return_value=pusher):
        updater.connect([], "nccl.so", **TIMEOUTS)
    out = updater.push("4")
    assert out["ok"] is False and "every server failed" in out["error"]
    assert consumed == [0, 1, 2] and rank["barriers"] == 1


def test_a_successful_push_reports_the_pusher_metrics(rank):
    pusher = MagicMock()
    pusher.connect.return_value = {}
    pusher.push.side_effect = lambda stream, version: {"version": version, "tensors": len(list(stream))}
    updater = VllmWeightUpdater(_stream([]))
    with patch.object(updater_module, "VllmPusher", return_value=pusher):
        updater.connect([], "nccl.so", **TIMEOUTS)
    assert updater.push("5") == {"version": "5", "tensors": 3, "ok": True}
