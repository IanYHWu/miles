"""Probe-set rows for sft_validation.probe_token_logprobs."""

import json

import pytest

from miles.backends.megatron_utils.sft_validation import _read_probe_rows


def _write(tmp_path, rows):
    path = tmp_path / "probe.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return str(path)


def test_rows_with_one_contiguous_target_are_read(tmp_path):
    rows = [{"tokens": [5, 6, 7], "loss_mask": [0, 1, 1]}, {"tokens": [1, 2], "loss_mask": [1, 1]}]
    assert _read_probe_rows(_write(tmp_path, rows)) == rows


@pytest.mark.parametrize(
    "row",
    [
        {"tokens": [1, 2, 3], "loss_mask": [0, 1]},  # length mismatch
        {"tokens": [1, 2], "loss_mask": [0, 0]},  # nothing to score
        {"tokens": [1, 2, 3], "loss_mask": [1, 0, 1]},  # not contiguous
        {"tokens": [1, 2], "loss_mask": [0, 2]},  # not 0/1
    ],
)
def test_invalid_rows_are_rejected(tmp_path, row):
    with pytest.raises(ValueError):
        _read_probe_rows(_write(tmp_path, [row]))


def test_an_empty_probe_set_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="empty"):
        _read_probe_rows(_write(tmp_path, []))


def test_the_probe_error_result_crosses_the_rpc_wire():
    """A failed probe answers {"error": ...}; under --worker-comm-backend rpc the result annotation must allow it."""
    from miles.backends.megatron_utils.actor import MegatronTrainRayActor
    from miles.utils.workers.rpc.common.metadata import collect_rpc_method_specs

    serializer = collect_rpc_method_specs(MegatronTrainRayActor)["probe_logprobs"].serializer
    for result in ({"error": "RuntimeError('x')"}, [[-0.5, -1.0]], None):
        assert serializer.decode_result(serializer.encode_result(result)) == result
