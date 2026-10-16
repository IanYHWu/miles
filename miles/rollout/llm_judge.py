"""Async HTTP judge with application-owned prompt builders and reply parsers.

The JSON config names a synchronous answer parser (returning ``answer`` and
``fault`` attributes) and judge factories exposing ``build_prompt`` and
``parse_reply``. Only these pure methods are called; HTTP is owned here.
This lets private projects retain their reward contract without importing a
second training framework or publishing their prompts in Miles.
"""

import asyncio
import json
import logging
import os
from functools import lru_cache
from pathlib import Path
from weakref import WeakKeyDictionary

import aiohttp

from miles.utils.function_registry import load_function

logger = logging.getLogger(__name__)
_states = WeakKeyDictionary()


class JudgeRuntime:
    def __init__(self, config, tokenizer=None):
        self.config = config
        self.tokenizer = tokenizer
        self.parser = load_function(config["answer_parser_path"], sync_required=True)
        self.judges = {
            name: load_function(spec["factory"], sync_required=True)(spec["config"])
            for name, spec in config["judges"].items()
        }
        self.semaphore = asyncio.Semaphore(config.get("max_concurrency", 64))
        if config.get("max_concurrency", 64) < 1:
            raise ValueError("judge max_concurrency must be positive")
        if config.get("max_input_tokens", 32768) < 1:
            raise ValueError("judge max_input_tokens must be positive")

    def prepare(self, sample):
        info = sample.metadata or {}
        kind = str(info.get("score_type") or "rubric").strip().lower()
        if kind not in self.judges:
            raise ValueError(f"No judge configured for score_type={kind!r}")
        label = sample.label
        rubric = label.get("ground_truth", "") if isinstance(label, dict) else str(label or "")
        reference = str(info.get("reference_answer") or "")
        if kind == "binary" and not reference.strip():
            raise ValueError("Binary judge requires metadata.reference_answer")
        if kind == "rubric" and not rubric.strip():
            raise ValueError("Rubric judge requires a nonempty rubric")
        readout = self.parser(sample.response)
        if readout.fault:
            return None, None, readout.fault
        answer = readout.answer
        if self.tokenizer is not None:
            ids = self.tokenizer.encode(answer, add_special_tokens=False)
            cap = self.config.get("max_input_tokens", 32768)
            if len(ids) > cap:
                answer = self.tokenizer.decode(ids[:cap], skip_special_tokens=False)
                answer += "\n\n[…response truncated for grader budget…]"
        problem = info.get("raw_problem")
        if not problem:
            raise ValueError("Judge requires metadata.raw_problem (not a rendered chat template)")
        judge = self.judges[kind]
        return judge, judge.build_prompt(str(problem), rubric, reference, answer), None

    async def chat(self, prompt):
        cfg = self.config
        key = os.environ.get(cfg.get("api_key_env", "LLM_JUDGE_API_KEY"))
        if not key:
            raise ValueError("Judge API credential environment variable is unset")
        payload = {
            "model": cfg["model"],
            "messages": [{"role": "user", "content": prompt}],
            "max_completion_tokens": cfg.get("max_output_tokens", 24576),
            "temperature": 1.0,
            "top_p": 1.0,
        }
        if cfg.get("reasoning_effort") is not None:
            payload["reasoning_effort"] = cfg["reasoning_effort"]
        delay = cfg.get("retry_initial_s", 5.0)
        attempts = 0
        async with self.semaphore:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=cfg.get("timeout_s", 900))
            ) as session:
                while True:
                    attempts += 1
                    try:
                        async with session.post(
                            cfg["endpoint_url"], json=payload, headers={"Authorization": f"Bearer {key}"}
                        ) as response:
                            if 400 <= response.status < 500 and response.status not in (408, 429):
                                # A permanent configuration failure must stop the job;
                                # unlike the reference recipe, never train on auth-error zeros.
                                raise PermissionError(
                                    f"Judge HTTP {response.status}; check endpoint/model/credentials"
                                )
                            response.raise_for_status()
                            body = await response.json()
                            return body["choices"][0]["message"].get("content") or ""
                    except (aiohttp.ClientError, asyncio.TimeoutError):
                        if not cfg.get("block_on_failure", True) and attempts >= cfg.get("max_attempts", 3):
                            raise
                        # Do not log response bodies, prompts or authorization headers.
                        logger.warning("Judge unavailable; retry %d in %.1fs", attempts, delay)
                        await asyncio.sleep(delay)
                        delay = min(delay * 2, cfg.get("retry_max_s", 60.0))

    async def score(self, sample):
        judge, prompt, fault = await asyncio.to_thread(self.prepare, sample)
        if fault:
            return {"score": 0.0, "judged": 0.0, "truncated": float(fault == "truncated")}
        parsed = judge.parse_reply(await self.chat(prompt))
        return {"score": float(parsed["reward"]), "judged": float(parsed["ok"]), "truncated": 0.0}


@lru_cache
def _load_config(path):
    return json.loads(Path(path).read_text())


async def reward(args, sample, **kwargs):
    if not args.llm_judge_config:
        raise ValueError("--llm-judge-config is required")
    loop_states = _states.setdefault(asyncio.get_running_loop(), {})
    key = (args.llm_judge_config, args.hf_checkpoint)
    if key not in loop_states:
        # Optional heavyweight dependency: load only when this reward is selected.
        from miles.utils.processing_utils import load_tokenizer

        tokenizer = load_tokenizer(args.hf_checkpoint, trust_remote_code=False)
        loop_states[key] = JudgeRuntime(_load_config(args.llm_judge_config), tokenizer)
    return await loop_states[key].score(sample)
