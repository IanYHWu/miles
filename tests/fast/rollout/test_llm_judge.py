import asyncio
from types import SimpleNamespace

import pytest
from aiohttp import web

from miles.rollout.llm_judge import ChatRequest, JudgeRuntime, reward
from miles.utils.function_registry import function_registry


class Adapter:
    def prepare(self, sample):
        if sample is None:
            return {"score": -1, "application_metric": 7}
        return ChatRequest(
            messages=[{"role": "user", "content": sample}],
            context=2,
            generation_kwargs={"temperature": 0.25},
        )

    def parse_reply(self, content, context):
        return {"score": float(content) * context}


@pytest.fixture
def config(monkeypatch):
    monkeypatch.setenv("LLM_JUDGE_API_KEY", "test-only")
    return {
        "model": "test",
        "endpoint_url": "unused",
        "generation_kwargs": {"temperature": 0.8, "max_completion_tokens": 32},
        "retry_initial_s": 0.001,
        "retry_max_s": 0.001,
    }


async def _with_server(handler, run):
    app = web.Application()
    app.router.add_post("/chat", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    try:
        await run(f"http://127.0.0.1:{port}/chat")
    finally:
        await runner.cleanup()


def test_adapter_owns_reward_and_context_without_dataset_assumptions(config):
    assert asyncio.run(JudgeRuntime(config, Adapter()).score(None)) == {"score": -1, "application_metric": 7}

    async def handler(request):
        assert request.headers["Authorization"] == "Bearer test-only"
        assert await request.json() == {
            "model": "test",
            "messages": [{"role": "user", "content": "arbitrary application text"}],
            "temperature": 0.25,
            "max_completion_tokens": 32,
        }
        return web.json_response({"choices": [{"message": {"content": "0.5"}}]})

    async def run(url):
        assert await JudgeRuntime({**config, "endpoint_url": url}, Adapter()).score("arbitrary application text") == {
            "score": 1
        }

    asyncio.run(_with_server(handler, run))
    assert config["generation_kwargs"]["temperature"] == 0.8


@pytest.mark.parametrize("status,expected_calls", [(400, 1), (401, 1), (403, 1), (408, 2), (429, 2), (500, 2)])
def test_retries_only_transient_failures_by_default(config, status, expected_calls):
    calls = []

    async def handler(request):
        calls.append(1)
        if len(calls) == 1:
            return web.Response(status=status)
        return web.json_response({"choices": [{"message": {"content": "1"}}]})

    async def run(url):
        runtime = JudgeRuntime({**config, "endpoint_url": url}, Adapter())
        if expected_calls == 1:
            with pytest.raises(RuntimeError, match=f"HTTP {status}"):
                await runtime.score("text")
        else:
            assert await runtime.score("text") == {"score": 2}
        assert len(calls) == expected_calls

    asyncio.run(_with_server(handler, run))


@pytest.mark.parametrize("status,extra", [(503, {}), (400, {"retry_bad_requests": True, "block_on_failure": True})])
def test_bounded_failures_and_private_diagnostics(config, status, extra, caplog, tmp_path):
    calls = []

    async def handler(request):
        calls.append(1)
        return web.json_response(
            {"error": {"code": "invalid_prompt", "message": "private-prompt-must-not-be-logged"}},
            status=status,
            headers={"x-request-id": "req_123"},
        )

    async def run(url):
        runtime = JudgeRuntime({**config, **extra, "endpoint_url": url, "error_log_dir": str(tmp_path)}, Adapter())
        with pytest.raises(RuntimeError, match="after 3 attempts") as error:
            await runtime.score("text")
        assert len(calls) == 3
        assert "private-prompt-must-not-be-logged" not in str(error.value) + caplog.text
        assert "invalid_prompt" in str(error.value)

    asyncio.run(_with_server(handler, run))
    files = list(tmp_path.glob("judge-error-*.json"))
    assert len(files) == 3
    assert all(path.stat().st_mode & 0o777 == 0o600 for path in files)


def test_cancellation_is_not_retried(config, monkeypatch):
    async def cancelled(self, request):
        raise asyncio.CancelledError()

    monkeypatch.setattr(JudgeRuntime, "chat", cancelled)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(JudgeRuntime(config, Adapter()).score("text"))


@pytest.mark.parametrize("kwargs", [{"max_concurrency": 0}, {"max_attempts": -1}, {"timeout_s": float("nan")}])
def test_rejects_invalid_limits(config, kwargs):
    with pytest.raises(ValueError):
        JudgeRuntime({**config, **kwargs}, Adapter())


def test_generation_overrides_cannot_change_transport_owned_fields(config):
    with pytest.raises(ValueError, match="cannot override"):
        JudgeRuntime({**config, "generation_kwargs": {"model": "another"}}, Adapter())
    runtime = JudgeRuntime(config, Adapter())
    with pytest.raises(ValueError, match="cannot override"):
        asyncio.run(runtime.chat(ChatRequest([], generation_kwargs={"stream": True})))


def test_reward_entrypoint_uses_configured_factory(config, tmp_path):
    import json

    path = tmp_path / "judge.json"
    path.write_text(json.dumps({**config, "adapter_factory": "test.adapter", "adapter_config": {"key": 1}}))
    calls = []

    def factory(adapter_config):
        calls.append(adapter_config)
        return Adapter()

    async def run():
        args = SimpleNamespace(llm_judge_config=str(path))
        assert (await reward(args, None))["score"] == -1
        assert (await reward(args, None))["score"] == -1

    with function_registry.temporary("test.adapter", factory):
        asyncio.run(run())
    assert calls == [{"key": 1}]
