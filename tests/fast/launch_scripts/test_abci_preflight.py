"""tools/abci/preflight.py builds the judge exactly as miles.rollout.llm_judge.reward does."""

import asyncio
import sys
from types import ModuleType

import pytest
from tools.abci import preflight

ROW = {
    "prompt": [{"role": "user", "content": "What is 1 + 1?"}],
    "reward_model": {"ground_truth": "2"},
    "extra_info": {"raw_problem": "What is 1 + 1?", "score_type": "binary"},
}


class Adapter:
    def __init__(self, config):
        self.config, self.prepared = config, []

    def prepare(self, sample):
        self.prepared.append(sample)
        assert sample.label == ROW["reward_model"] and sample.metadata == ROW["extra_info"]
        return preflight.ChatRequest(messages=[{"role": "user", "content": sample.response}], context="ctx")

    def parse_reply(self, content, context):
        return {self.config["reward_key"]: 1.0}


@pytest.fixture
def config(monkeypatch):
    module = ModuleType("abci_preflight_adapter")
    module.make = Adapter
    monkeypatch.setitem(sys.modules, "abci_preflight_adapter", module)
    return {
        "endpoint_url": "https://judge.invalid/v1/chat/completions",
        "model": "judge",
        "adapter_factory": "abci_preflight_adapter.make",
        "adapter_config": {"reward_key": "score"},
    }


def _probe(runtime, replies):
    async def chat(request):
        return replies.pop(0)

    runtime.chat = chat
    return asyncio.run(preflight._probe_judge(runtime, ROW))


def test_the_runtime_uses_the_configured_adapter_factory(config):
    runtime, adapter = preflight._runtime(config)
    assert isinstance(adapter, Adapter) and adapter.config == config["adapter_config"]
    _probe(runtime, ["OK", "judged"])
    assert adapter.prepared[-1].prompt == ROW["prompt"]


def test_an_empty_live_reply_fails_the_probe(config):
    runtime, _ = preflight._runtime(config)
    with pytest.raises(RuntimeError, match="empty reply"):
        _probe(runtime, ["  ", "judged"])


def test_a_reward_without_the_trained_key_fails_the_probe(config):
    config["adapter_config"] = {"reward_key": "reward"}
    runtime, _ = preflight._runtime(config)
    with pytest.raises(RuntimeError, match="'score'"):
        _probe(runtime, ["OK", "judged"])
