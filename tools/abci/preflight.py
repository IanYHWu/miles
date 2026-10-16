"""Read-only checks of the environment and all judge-scored input rows."""

import argparse
import asyncio
import json
import os
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pyarrow.parquet as pq
import torch
from scripts.run_qwen3_5_judge import ScriptArgs

from miles.rollout.llm_judge import JudgeRuntime
from miles.utils.processing_utils import load_tokenizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--judge-config", required=True)
    p.add_argument("--model-name", default="Qwen3.5-9B")
    p.add_argument("--probe-judge", action="store_true", help="Make one live judge call on a known-answer fixture")
    args = p.parse_args()
    model = Path(args.model_dir) / args.model_name
    tokenizer = load_tokenizer(str(model), local_files_only=True)
    cfg = json.loads(Path(args.judge_config).read_text())
    runtime = JudgeRuntime(cfg, tokenizer)
    prompt_limit = ScriptArgs(smoke=True, model_name=args.model_name).prompt_length
    for name in ("train", "val"):
        rows = pq.read_table(Path(args.data_dir) / f"{name}.parquet").to_pylist()
        if not rows:
            raise ValueError(f"Empty {name} dataset")
        max_length = 0
        for row in rows:
            ids = tokenizer.apply_chat_template(
                row["prompt"], tokenize=True, add_generation_prompt=True, return_dict=False
            )
            max_length = max(max_length, len(ids))
            runtime.prepare(
                SimpleNamespace(response="</think>preflight", label=row["reward_model"], metadata=row["extra_info"])
            )
        if max_length > prompt_limit:
            raise ValueError(f"{name} has prompt length {max_length} > {prompt_limit}; do not silently filter data")
        print(
            name,
            "rows",
            len(rows),
            "max_prompt_tokens",
            max_length,
            "score_types",
            dict(Counter(r["extra_info"]["score_type"] for r in rows)),
            flush=True,
        )
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
        score = asyncio.run(
            asyncio.wait_for(
                runtime.score(
                    SimpleNamespace(
                        response="</think>2",
                        label={"ground_truth": "Award 1 point for the answer 2."},
                        metadata={"raw_problem": "What is 1 + 1?", "score_type": "binary", "reference_answer": "2"},
                    )
                ),
                timeout=60,
            )
        )
        if score["score"] != 1.0 or score["judged"] != 1.0:
            raise RuntimeError(f"Known-answer live judge probe failed: {score}")
        print("Live judge probe passed", flush=True)


if __name__ == "__main__":
    main()
