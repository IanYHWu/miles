"""OpenAI-compatible judge transport with application-owned scoring adapters."""

import asyncio
import json
import logging
import os
import re
import tempfile
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol
from weakref import WeakKeyDictionary

import aiohttp

from miles.utils.function_registry import load_function

logger = logging.getLogger(__name__)
_states = WeakKeyDictionary()


@dataclass(frozen=True)
class ChatRequest:
    """Adapter-built messages, generation overrides, and reply interpretation context."""

    messages: list[dict[str, Any]]
    context: Any = None
    generation_kwargs: dict[str, Any] = field(default_factory=dict)


class JudgeAdapter(Protocol):
    """Own dataset fields, prompt construction, parsing, and reward policy.

    Methods are synchronous and may run concurrently in worker threads. Returning
    a reward dictionary from prepare skips HTTP entirely, for example for an
    answer that the application considers invalid.
    """

    def prepare(self, sample: Any) -> ChatRequest | dict[str, Any]: ...

    def parse_reply(self, content: str, context: Any) -> dict[str, Any]: ...


class JudgeRuntime:
    def __init__(self, config: dict, adapter: JudgeAdapter):
        self._config = dict(config)
        self._adapter = adapter
        for key in ("endpoint_url", "model"):
            if not self._config.get(key):
                raise ValueError(f"Judge requires {key}")
        for key, default in (("max_concurrency", 64), ("max_attempts", 3)):
            value = self._config.get(key, default)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"Judge {key} must be a positive integer")
        for key, default in (("timeout_s", 900), ("retry_initial_s", 1.0), ("retry_max_s", 60.0)):
            value = self._config.get(key, default)
            if not isinstance(value, (int, float)) or not 0 < value < float("inf"):
                raise ValueError(f"Judge {key} must be finite and positive")
        self._generation_kwargs = dict(self._config.get("generation_kwargs", {}))
        _validate_generation_kwargs(self._generation_kwargs)
        self._semaphore = asyncio.Semaphore(self._config.get("max_concurrency", 64))

    async def score(self, sample: Any) -> dict[str, Any]:
        prepared = await asyncio.to_thread(self._adapter.prepare, sample)
        if isinstance(prepared, dict):
            return prepared
        if not isinstance(prepared, ChatRequest):
            raise TypeError("Judge adapter.prepare must return ChatRequest or a reward dictionary")
        content = await self.chat(prepared)
        reward = await asyncio.to_thread(self._adapter.parse_reply, content, prepared.context)
        if not isinstance(reward, dict):
            raise TypeError("Judge adapter.parse_reply must return a reward dictionary")
        return reward

    async def chat(self, request: ChatRequest) -> str:
        cfg = self._config
        key = os.environ.get(cfg.get("api_key_env", "LLM_JUDGE_API_KEY"))
        if not key:
            raise ValueError("Judge API credential environment variable is unset")
        _validate_generation_kwargs(request.generation_kwargs)
        # Per-request generation settings override configured defaults. Model and
        # messages have separate owners and cannot be overridden through kwargs.
        payload = {**self._generation_kwargs, **request.generation_kwargs}
        payload.update(model=cfg["model"], messages=request.messages)
        async with self._semaphore:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=cfg.get("timeout_s", 900))
            ) as session:
                return await self._chat_with_retries(session, payload, key)

    async def _chat_with_retries(self, session, payload, key):
        cfg = self._config
        delay = cfg.get("retry_initial_s", 1.0)
        attempts = bad_requests = 0
        while True:
            attempts += 1
            try:
                async with session.post(
                    cfg["endpoint_url"], json=payload, headers={"Authorization": f"Bearer {key}"}
                ) as response:
                    if response.status >= 400:
                        details = await _http_error_summary(response, cfg.get("error_log_dir"))
                        transient = response.status in (408, 429) or 500 <= response.status < 600
                        if response.status == 400 and cfg.get("retry_bad_requests", False):
                            bad_requests += 1
                            if bad_requests >= cfg.get("max_attempts", 3):
                                raise RuntimeError(f"Judge HTTP 400 after {bad_requests} attempts; {details}")
                            transient = True
                        if not transient:
                            raise RuntimeError(f"Judge HTTP {response.status}; {details}")
                        # Do not include the response body or request headers in
                        # exceptions emitted for retryable HTTP failures either.
                        failure = f"HTTP {response.status}; {details}"
                    else:
                        body = await response.json()
                        return _reply_content(body)
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                failure = type(exc).__name__
            if not cfg.get("block_on_failure", False) and attempts >= cfg.get("max_attempts", 3):
                raise RuntimeError(f"Judge unavailable after {attempts} attempts: {failure}") from None
            logger.warning("Judge unavailable (%s); retry %d in %.1fs", failure, attempts, delay)
            await asyncio.sleep(delay)
            delay = min(delay * 2, cfg.get("retry_max_s", 60.0))


def _validate_generation_kwargs(kwargs):
    if set(kwargs) & {"model", "messages", "stream"}:
        raise ValueError("Judge generation_kwargs cannot override model, messages, or stream")


def _reply_content(body):
    try:
        content = body["choices"][0]["message"].get("content")
    except (KeyError, IndexError, TypeError, AttributeError):
        raise ValueError("Judge response must contain choices[0].message") from None
    if content is not None and not isinstance(content, str):
        raise ValueError("Judge response content must be text or null")
    return content or ""


async def _http_error_summary(response, error_log_dir=None):
    # Provider messages can echo prompts; only bounded identifiers reach logs.
    try:
        body = await response.json(content_type=None)
    except (ValueError, aiohttp.ContentTypeError):
        body = {}
    if error_log_dir:
        directory = Path(error_log_dir)
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        with tempfile.NamedTemporaryFile(
            mode="w", dir=directory, prefix="judge-error-", suffix=".json", delete=False
        ) as output:
            json.dump({"status": response.status, "body": body}, output)
    error = body.get("error", {}) if isinstance(body, dict) else {}
    if not isinstance(error, dict):
        error = {}
    values = {name: error.get(name) for name in ("type", "code", "param")}
    values["request_id"] = response.headers.get("x-request-id")
    return ", ".join(
        f"{name}={value if isinstance(value, str) and re.fullmatch(r'[A-Za-z0-9_.-]{1,100}', value) else 'unavailable'}"
        for name, value in values.items()
    )


@lru_cache
def _load_config(path):
    return json.loads(Path(path).read_text())


async def reward(args, sample, **kwargs):
    """Reward entrypoint; adapter_factory(config) supplies application behavior."""
    if not args.llm_judge_config:
        raise ValueError("--llm-judge-config is required")
    loop_states = _states.setdefault(asyncio.get_running_loop(), {})
    config_path = str(Path(args.llm_judge_config).resolve())
    if config_path not in loop_states:
        config = _load_config(config_path)
        factory = load_function(config["adapter_factory"], sync_required=True)
        loop_states[config_path] = JudgeRuntime(config, factory(config.get("adapter_config", {})))
    return await loop_states[config_path].score(sample)
