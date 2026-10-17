import asyncio
from types import SimpleNamespace

import pytest
from aiohttp import web

from miles.rollout.llm_judge import JudgeRuntime
from miles.utils.function_registry import function_registry


class FakeJudge:
    def __init__(self, config):
        pass

    def build_prompt(self, problem, rubric, reference, answer):
        return f"{problem}|{rubric}|{reference}|{answer}"

    def parse_reply(self, content):
        return {"reward": 0.5 if content == "half" else 0.0, "ok": content == "half"}


def read_answer(text):
    return SimpleNamespace(answer=text.partition("</think>")[2], fault=None if "</think>" in text else "truncated")


@pytest.fixture
def config():
    with (
        function_registry.temporary("test_judge.factory", FakeJudge),
        function_registry.temporary("test_judge.read", read_answer),
    ):
        yield {
            "model": "test",
            "endpoint_url": "unused",
            "answer_parser_path": "test_judge.read",
            "judges": {k: {"factory": "test_judge.factory", "config": {}} for k in ("rubric", "binary")},
            "retry_initial_s": 0.001,
            "retry_max_s": 0.001,
        }


def sample(response="hidden</think>answer", kind="rubric"):
    return SimpleNamespace(
        response=response,
        label={"ground_truth": "rubric"},
        metadata={"raw_problem": "question", "score_type": kind, "reference_answer": "gold"},
    )


def test_prepare_routes_and_never_sends_thinking(config):
    runtime = JudgeRuntime(config)
    for kind in ("rubric", "binary"):
        _, prompt, fault = runtime.prepare(sample(kind=kind))
        assert prompt == "question|rubric|gold|answer"
        assert fault is None
    assert asyncio.run(runtime.score(sample("unclosed thinking")))["truncated"] == 1
    with pytest.raises(ValueError, match="score_type"):
        runtime.prepare(sample(kind="typo"))


@pytest.mark.parametrize("status", [400, 429, 500, 401])
def test_http_retry_and_permanent_failure(config, monkeypatch, status):
    monkeypatch.setenv("LLM_JUDGE_API_KEY", "test-only")

    async def run():
        calls = []

        async def handler(request):
            calls.append(await request.json())
            if len(calls) == 1:
                return web.Response(status=status)
            return web.json_response({"choices": [{"message": {"content": "half"}}]})

        app = web.Application()
        app.router.add_post("/chat", handler)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        try:
            runtime = JudgeRuntime({**config, "endpoint_url": f"http://127.0.0.1:{port}/chat"})
            if status == 401:
                with pytest.raises(PermissionError):
                    await runtime.score(sample())
                assert len(calls) == 1
            else:
                assert (await runtime.score(sample()))["score"] == 0.5
                assert len(calls) == 2
                assert "max_completion_tokens" in calls[0]
        finally:
            await runner.cleanup()

    asyncio.run(run())


def test_cancelled_judge_is_not_retried(config, monkeypatch):
    async def cancel(self, prompt):
        raise asyncio.CancelledError()

    monkeypatch.setattr(JudgeRuntime, "chat", cancel)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(JudgeRuntime(config).score(sample()))


def test_bad_requests_are_bounded_and_diagnostics_do_not_echo_prompts(config, monkeypatch, caplog, tmp_path):
    monkeypatch.setenv("LLM_JUDGE_API_KEY", "test-only")

    async def run():
        calls = 0

        async def handler(request):
            nonlocal calls
            calls += 1
            return web.json_response(
                {
                    "error": {
                        "code": "invalid_prompt",
                        "type": "invalid_request_error",
                        "message": "private-prompt-must-not-be-logged",
                    }
                },
                status=400,
                headers={"x-request-id": "req_123"},
            )

        app = web.Application()
        app.router.add_post("/chat", handler)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        try:
            runtime = JudgeRuntime(
                {
                    **config,
                    "block_on_failure": True,
                    "error_log_dir": str(tmp_path),
                    "endpoint_url": f"http://127.0.0.1:{port}/chat",
                }
            )
            with pytest.raises(RuntimeError, match="HTTP 400 after 3 attempts") as error:
                await runtime.score(sample())
            assert calls == 3
            files = list(tmp_path.glob("judge-error-*.json"))
            assert len(files) == 3
            assert all(path.stat().st_mode & 0o777 == 0o600 for path in files)
            assert "private-prompt-must-not-be-logged" in files[0].read_text()
            assert "invalid_prompt" in str(error.value) and "req_123" in str(error.value)
            assert "private-prompt-must-not-be-logged" not in str(error.value) + caplog.text
        finally:
            await runner.cleanup()

    asyncio.run(run())
