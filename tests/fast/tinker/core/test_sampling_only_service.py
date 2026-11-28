"""A gateway that trains nothing samples the base model and refuses training models and sampler checkpoints."""

import asyncio
from contextlib import suppress

import pytest
from tests.fast.tinker.harness import await_settled, make_service, model_payload

from miles.tinker.core.future import DONE
from miles.tinker.core.types import UserInputError


@pytest.fixture
async def sampling_service(tmp_path):
    gateway = make_service(tmp_path, train_mode="none", n_slots=0)
    run_task = asyncio.create_task(gateway.run())
    try:
        yield gateway
    finally:
        run_task.cancel()
        with suppress(asyncio.CancelledError):
            await run_task


def _sample_payload(model_path: str | None) -> dict:
    return {
        "model_path": model_path,
        "num_samples": 1,
        "prompt_tokens": [1, 2],
        "sampling_params": {"max_tokens": 4},
        "prompt_logprobs": False,
        "topk_prompt_logprobs": 0,
    }


async def test_a_gateway_that_trains_nothing_samples_the_base_model(sampling_service):
    request_id, _ = sampling_service.submit_sample("tenant", _sample_payload(None))
    assert (await await_settled(sampling_service, "tenant", request_id)).state == DONE
    assert [call["lora_name"] for call in sampling_service.backend.named("sample")] == [None]


def test_a_gateway_that_trains_nothing_creates_no_training_model(sampling_service):
    with pytest.raises(UserInputError, match="trains nothing"):
        sampling_service.create_model("tenant", model_payload(sampling_service, lora_config={"rank": 8}))


def test_a_gateway_that_trains_nothing_refuses_a_sampler_checkpoint(sampling_service):
    with pytest.raises(UserInputError, match="samples only the base model"):
        sampling_service.submit_sample("tenant", _sample_payload("tinker://model-abc/sampler_weights/1"))
