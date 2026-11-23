"""Variable-length native requests and evidence checks for the 9B async smoke.

Rewards remain synthetic. No response text, dataset content, or credentials are
written to the evidence file. A failed coverage assertion fails the smoke.
"""

import json
import math
import time
from dataclasses import asdict, replace
from datetime import datetime
from itertools import pairwise
from pathlib import Path

from miles.rollout.generate_hub.single_turn import generate as single_turn_generate


def _record(args, event):
    path = Path(args.save).parent / "async-evidence.jsonl"
    with path.open("a") as stream:
        stream.write(json.dumps({"time": time.time(), **event}) + "\n")


def _sample_evidence(sample):
    spans = list(sample.all_weight_version_spans)
    assert spans, "No token weight versions reported"
    assert sum(s.abs_end - s.abs_start for s in spans) == sample.response_length
    assert len(sample.rollout_log_probs) == sample.response_length
    assert all(math.isfinite(x) for x in sample.rollout_log_probs)
    return {
        "index": sample.index,
        "response_length": sample.response_length,
        "versions": sorted({int(s.version) for s in spans}),
        "spans": [asdict(s) for s in spans],
    }


async def generate(input):
    # Siblings share a length. Long requests stay inside one native SGLang call,
    # so mixed versions exercise in-place pause/resume, not application retries.
    group = input.sample.index // input.args.n_samples_per_prompt
    length = (4096, 1024, 128, 128)[group % 4]
    started = time.time()
    output = await single_turn_generate(
        replace(input, sampling_params={**input.sampling_params, "max_new_tokens": length, "ignore_eos": True})
    )
    _record(input.args, {"type": "completed", "started": started, **_sample_evidence(output.samples)})
    return output


def log_rollout_data(rollout_id, args, samples, rollout_extra_metrics, rollout_time):
    _record(
        args,
        {
            "type": "selected",
            "rollout_id": rollout_id,
            "samples": [_sample_evidence(s) for s in samples],
            "metrics": rollout_extra_metrics,
            "wait_seconds": rollout_time,
        },
    )
    return False


def before_train(args, rollout_id, step_id, model, optimizer, opt_param_scheduler):
    import torch.distributed as dist

    if dist.get_rank() == 0:
        path = Path(args.save).parent / "async-train-starts.jsonl"
        with path.open("a") as stream:
            stream.write(json.dumps({"rollout_id": rollout_id, "time": time.time()}) + "\n")


def validate(run_dir):
    import torch

    from scripts.run_qwen3_5_judge import ScriptArgs

    config = ScriptArgs(async_smoke=True)
    records = [json.loads(line) for line in (run_dir / "async-evidence.jsonl").read_text().splitlines()]
    selected = [r for r in records if r["type"] == "selected"]
    assert [r["rollout_id"] for r in selected] == list(range(config.num_rollout))
    mixed = [s for r in selected for s in r["samples"] if len(s["versions"]) > 1]
    assert mixed, "No trajectory spanning a weight publication reached training"
    for r in selected:
        assert r["metrics"]["rollout/fully_async/max_staleness"] <= config.max_weight_staleness
        assert r["metrics"]["rollout/fully_async/weight_version_sample_coverage"] == 1
    norms = [float(torch.load(run_dir / f"grad-{step}.pt", weights_only=False)) for step in range(config.num_rollout)]
    assert all(math.isfinite(n) and n > 0 for n in norms), norms
    checkpoint = run_dir / "checkpoints"
    assert int((checkpoint / "latest_checkpointed_iteration.txt").read_text()) == config.num_rollout - 1
    events = [
        json.loads(line) for path in (checkpoint / "events").glob("*.jsonl") for line in path.read_text().splitlines()
    ]
    starts = [json.loads(line) for line in (run_dir / "async-train-starts.jsonl").read_text().splitlines()]
    ends = {
        e["rollout_id"]: datetime.fromisoformat(e["timestamp"]).timestamp()
        for e in events
        if e.get("type") == "train_group_step_end" and e["role"] == "actor"
    }
    overlapped = [
        r
        for r in records
        if r["type"] == "completed" and any(s["time"] < r["time"] < ends[s["rollout_id"]] for s in starts)
    ]
    assert overlapped, "No native generation completed while the trainer was running"
    checksums = [e for e in events if e.get("type") == "inference_engine_weight_checksum"]
    assert len(checksums) == 1 + config.num_rollout // config.update_interval
    for before, after in pairwise(checksums):
        assert before["engine_checksums"] != after["engine_checksums"], "Published weights did not change"
    result = {
        "passed": True,
        "updates": config.num_rollout,
        "gradient_norms": norms,
        "mixed_version_samples_trained": len(mixed),
        "generations_completed_during_training": len(overlapped),
        "weight_publications_after_initial": len(checksums) - 1,
        "stale_groups_filtered": sum(r["metrics"]["rollout/fully_async/stale_groups_filtered"] for r in selected),
        "max_selected_staleness": max(r["metrics"]["rollout/fully_async/max_staleness"] for r in selected),
        "synthetic_rewards": True,
    }
    (run_dir / "async-result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    import sys

    validate(Path(sys.argv[1]))
