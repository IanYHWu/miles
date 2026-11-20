"""A full fine-tuning gateway trains one model at a time, without adapters, and its checkpoints say so."""

import asyncio
import json
import os
from contextlib import suppress

import pytest
from tests.fast.tinker.harness import await_settled, created_model, datum, fb_payload, make_service, model_payload

from miles.tinker.core.future import DONE, FAILED
from miles.tinker.core.types import UserInputError
from miles.tinker.core.utils import resolve_checkpoint_dir


@pytest.fixture
async def full_service(tmp_path):
    gateway = make_service(tmp_path, train_mode="full", n_slots=1, trains_unembed=True)
    run_task = asyncio.create_task(gateway.run())
    try:
        yield gateway
    finally:
        run_task.cancel()
        with suppress(asyncio.CancelledError):
            await run_task


async def _saved_checkpoint(service, model_id: str) -> str:
    saved = service.submit(
        "tenant", "save_state", {"model_id": model_id, "seq_id": 1, "name": "ck", "overwrite": False}
    )
    return (await await_settled(service, "tenant", saved)).result["path"]


def _meta_path(service, model_id: str) -> str:
    return os.path.join(resolve_checkpoint_dir(service.config.checkpoint_root, model_id, "weights", "ck"), "META.json")


def _rewrite_meta(service, model_id: str, **changes) -> None:
    meta_path = _meta_path(service, model_id)
    meta = json.loads(open(meta_path).read())
    for key, value in changes.items():
        if value is None:
            del meta[key]
        else:
            meta[key] = value
    open(meta_path, "w").write(json.dumps(meta))


async def test_a_full_finetune_model_takes_and_ignores_the_sdk_lora_config(full_service):
    lora_config = {"rank": 64, "train_unembed": False, "seed": 3}
    request_id, model_id = full_service.create_model("tenant", model_payload(full_service, lora_config=lora_config))

    future = await await_settled(full_service, "tenant", request_id)
    assert future.result == {"op": "create_model", "model_id": model_id}
    [load] = full_service.backend.named("load_slot")
    assert (load["slot"], load["rank"], load["alpha"], load["ckpt_path"]) == (0, 0, 0.0, None)


async def test_a_full_finetune_gateway_trains_one_model_at_a_time(full_service):
    await created_model(full_service)

    with pytest.raises(UserInputError, match="one model at a time"):
        full_service.create_model("other", model_payload(full_service, "other"))


async def test_a_full_finetune_checkpoint_names_its_train_mode_and_resumes(full_service):
    model_id = await created_model(full_service)
    path = await _saved_checkpoint(full_service, model_id)

    meta = json.loads(open(_meta_path(full_service, model_id)).read())
    assert (meta["train_mode"], meta["lora_rank"], meta["lora_alpha"]) == ("full", 0, 0.0)
    # the SDK only resumes LoRA training clients, so the checkpoint says it is one, of rank 0
    assert full_service.weights_info("tenant", path) == {
        "base_model": "base",
        "is_lora": True,
        "lora_rank": 0,
        "train_attn": True,
        "train_mlp": True,
        "train_unembed": True,
    }
    loaded = full_service.submit(
        "tenant", "load_state", {"model_id": model_id, "seq_id": 2, "path": path, "optimizer": True}
    )
    assert (await await_settled(full_service, "tenant", loaded)).state == DONE
    assert full_service.backend.named("load_slot")[-1]["load_optimizer"] is True


async def test_an_adapter_checkpoint_does_not_load_into_a_full_finetune_model(full_service):
    model_id = await created_model(full_service)
    path = await _saved_checkpoint(full_service, model_id)
    _rewrite_meta(full_service, model_id, train_mode="lora", lora_rank=8, lora_alpha=16.0)

    loaded = full_service.submit(
        "tenant", "load_state", {"model_id": model_id, "seq_id": 2, "path": path, "optimizer": True}
    )
    future = await await_settled(full_service, "tenant", loaded)
    assert (future.state, future.error_category) == (FAILED, "user") and "train_mode" in future.error
    assert not full_service.backend.named("load_slot")[1:], "nothing may touch the weights on a mismatch"


async def test_full_finetune_metadata_with_an_adapter_shape_is_refused(full_service):
    model_id = await created_model(full_service)
    path = await _saved_checkpoint(full_service, model_id)
    _rewrite_meta(full_service, model_id, lora_rank=8)

    with pytest.raises(UserInputError, match="invalid full fine-tuning metadata"):
        full_service.weights_info("tenant", path)


async def test_a_checkpoint_naming_no_train_mode_holds_an_adapter(service):
    model_id = await created_model(service)
    path = await _saved_checkpoint(service, model_id)
    _rewrite_meta(service, model_id, train_mode=None)

    assert service.weights_info("tenant", path)["lora_rank"] == 8
    loaded = service.submit(
        "tenant", "load_state", {"model_id": model_id, "seq_id": 2, "path": path, "optimizer": True}
    )
    assert (await await_settled(service, "tenant", loaded)).state == DONE


def _sample_payload(model_path: str | None) -> dict:
    return {
        "model_path": model_path,
        "num_samples": 1,
        "prompt_tokens": [1, 2],
        "sampling_params": {"max_tokens": 4},
        "prompt_logprobs": False,
        "topk_prompt_logprobs": 0,
    }


async def _sampler_save(service, model_id: str, seq_id: int) -> str:
    saved = service.submit("tenant", "save_weights_for_sampler", {"model_id": model_id, "seq_id": seq_id})
    return (await await_settled(service, "tenant", saved)).result["path"]


async def test_full_finetune_engines_serve_only_the_latest_sampler_save(full_service):
    """The engines hold one copy of the weights, so no sampler may be served another save's, or the base's."""
    full_service.submit_sample("tenant", _sample_payload(None))  # before any push the engines hold the base

    model_id = await created_model(full_service)
    first = await _sampler_save(full_service, model_id, 1)
    second = await _sampler_save(full_service, model_id, 2)

    request_id, _ = full_service.submit_sample("tenant", _sample_payload(second))
    assert (await await_settled(full_service, "tenant", request_id)).state == DONE
    for stale in (first, None):
        with pytest.raises(UserInputError, match="serves only its latest sampler save"):
            full_service.submit_sample("tenant", _sample_payload(stale))


async def test_a_new_full_finetune_model_samples_nothing_before_its_own_push(full_service):
    model_id = await created_model(full_service)
    await _sampler_save(full_service, model_id, 1)
    full_service.backend.fail_next = {"error": "batch rejected"}  # a failed batch closes the model
    closing = full_service.submit("tenant", "forward_backward", fb_payload(model_id, 2, [datum()]))
    await await_settled(full_service, "tenant", closing)
    assert model_id not in full_service.models

    await created_model(full_service, tenant="other")
    with pytest.raises(UserInputError, match="serves only its latest sampler save"):
        full_service.submit_sample("other", _sample_payload(None))
