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
    with function_registry.temporary("test_judge.factory", FakeJudge), function_registry.temporary(
        "test_judge.read", read_answer
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


@pytest.mark.parametrize("status", [429, 500, 401])
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
