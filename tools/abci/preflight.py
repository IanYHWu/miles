"""Read-only checks of the environment, the judge adapter and every judge-scored input row."""

import argparse
import asyncio
import json
import os
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
import torch
from scripts.run_qwen3_5_judge import ScriptArgs

from miles.rollout.llm_judge import ChatRequest, JudgeRuntime
from miles.utils.function_registry import load_function
from miles.utils.processing_utils import load_tokenizer
from miles.utils.types import Sample

# The reward key scripts/run_qwen3_5_judge.py trains on.
REWARD_KEY = "score"


def _runtime(config):
    """The same JudgeRuntime miles.rollout.llm_judge.reward builds from this config."""
    factory = load_function(config["adapter_factory"], sync_required=True)
    adapter = factory(config.get("adapter_config", {}))
    return JudgeRuntime(config, adapter), adapter


def _sample(row, response):
    return Sample(prompt=row["prompt"], response=response, label=row["reward_model"], metadata=row["extra_info"])


async def _probe_judge(runtime, row):
    """One live request through the transport, then one through the adapter on a real row."""
    reply = await asyncio.wait_for(
        runtime.chat(ChatRequest(messages=[{"role": "user", "content": "Reply with the single word OK."}])),
        timeout=120,
    )
    if not reply.strip():
        raise RuntimeError("Live judge probe returned an empty reply")
    reward = await asyncio.wait_for(runtime.score(_sample(row, "</think>preflight")), timeout=600)
    if REWARD_KEY not in reward:
        raise RuntimeError(f"Judge adapter reward has no {REWARD_KEY!r}: {sorted(reward)}")
    print(f"Live judge probe passed ({REWARD_KEY}={reward[REWARD_KEY]})", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--judge-config", required=True)
    p.add_argument("--model-name", default="Qwen3.5-9B")
    p.add_argument("--probe-judge", action="store_true", help="Send live requests through the configured judge")
    args = p.parse_args()
    model = Path(args.model_dir) / args.model_name
    tokenizer = load_tokenizer(str(model), local_files_only=True)
    cfg = json.loads(Path(args.judge_config).read_text())
    runtime, adapter = _runtime(cfg)
    prompt_limit = ScriptArgs(smoke=True, model_name=args.model_name).prompt_length
    first_row = None
    for name in ("train", "val"):
        rows = pq.read_table(Path(args.data_dir) / f"{name}.parquet").to_pylist()
        if not rows:
            raise ValueError(f"Empty {name} dataset")
        first_row = first_row or rows[0]
        max_length = 0
        for row in rows:
            ids = tokenizer.apply_chat_template(
                row["prompt"], tokenize=True, add_generation_prompt=True, return_dict=False
            )
            max_length = max(max_length, len(ids))
            # The adapter validates the row's fields without any HTTP call.
            adapter.prepare(_sample(row, "</think>preflight"))
        if max_length > prompt_limit:
            raise ValueError(f"{name} has prompt length {max_length} > {prompt_limit}; do not silently filter data")
        score_types = Counter(str((r.get("extra_info") or {}).get("score_type")) for r in rows)
        print(name, "rows", len(rows), "max_prompt_tokens", max_length, "score_types", dict(score_types), flush=True)
    print("torch", torch.__version__, "CUDA", torch.version.cuda, "GPUs", torch.cuda.device_count(), flush=True)
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA device accessible")
    x = torch.ones(8, device="cuda")
    assert x.sum().item() == 8
    import megatron.core
    import sglang
    import transformer_engine

    print("sglang", sglang.__version__, "TE", transformer_engine.__version__, "Megatron", megatron.core.__file__)
    if not os.environ.get(cfg.get("api_key_env", "LLM_JUDGE_API_KEY")):
        raise ValueError("Judge credential missing")
    if args.probe_judge:
        asyncio.run(_probe_judge(runtime, first_row))


if __name__ == "__main__":
    main()
