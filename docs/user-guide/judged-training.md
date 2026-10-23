---
title: Judged training and CISPO
description: Use an application owned reward adapter with an external chat judge and optionally train with the token level CISPO objective.
---

Miles supplies an asynchronous OpenAI-compatible chat transport. The application supplies dataset interpretation, prompt construction, answer extraction, reply parsing, and the reward policy. No dataset column names or rubric conventions are required by the transport.

## Judge adapter

Select `--custom-rm-path miles.rollout.llm_judge.reward --llm-judge-config /config/judge.json`. The configuration names an importable adapter factory:

```json
{
  "endpoint_url": "https://provider.example/v1/chat/completions",
  "model": "judge-model",
  "api_key_env": "LLM_JUDGE_API_KEY",
  "adapter_factory": "my_project.judge.make_adapter",
  "adapter_config": {},
  "max_concurrency": 32,
  "max_attempts": 3,
  "generation_kwargs": {"temperature": 0, "max_completion_tokens": 1024}
}
```

The factory takes `adapter_config` and returns an object with two synchronous, thread-safe methods:

```python
from miles.rollout.llm_judge import ChatRequest

class Adapter:
    def prepare(self, sample):
        messages, context = build_application_prompt(sample)
        return ChatRequest(messages=messages, context=context)

    def parse_reply(self, content, context):
        return score_application_reply(content, context)
```

`prepare` may instead return a reward dictionary to skip HTTP, for example when application-specific answer validation fails. `parse_reply` returns the reward dictionary expected by the configured reward keys. The transport does not convert failures into a fabricated score. Tokenization, truncation, and whether a null reply earns a reward are application decisions.

Per-request `ChatRequest.generation_kwargs` override configured generation defaults. They cannot replace `model`, `messages`, or enable streaming. The model and endpoint are runtime settings; messages belong to the adapter. Adapter methods execute in worker threads and can be called concurrently.

## Failure handling

HTTP 408/429/5xx and transport timeouts/errors retry with exponential backoff. The default attempt limit is finite. Set `block_on_failure` explicitly to keep retrying transient failures indefinitely. Other HTTP failures are permanent by default; `retry_bad_requests` opts into a bounded retry of HTTP 400 even when transient failures block indefinitely. Cancellation propagates.

`timeout_s`, `retry_initial_s`, and `retry_max_s` control the request timeout and retry delay. Credentials are read from the named environment variable. Logs contain bounded provider identifiers rather than request/response bodies. An optional `error_log_dir` stores response bodies in mode-0600 diagnostic files; those files can contain application data and belong in an application-controlled directory.

Applications with existing configuration formats can construct `JudgeRuntime(config, adapter)` directly inside their own reward entrypoint.

## CISPO objective

`--policy-loss-type cispo` selects token-level clipped importance-sampling policy optimization. The default remains `ppo`. With `r = exp(logp_current - logp_old)`, the per-token loss is:

```text
- stop_gradient(clip(r, 1 - eps_clip, 1 + eps_clip_high))
  * stop_gradient(advantage) * logp_current
```

The log ratio is bounded before exponentiation. Clipping arguments are offsets from one, following the existing CLI convention; `eps_clip_high` defaults to `eps_clip`. Existing masking, distributed reductions, and loss normalization remain in charge of aggregating token losses. CISPO requires `policy_loss`, rejects GSPO sequence ratios and dual clipping, and imposes no dataset, judge, staleness, or learning-rate settings.

## Model and checkpoint support

The Qwen3.5 model definitions include `qwen3.5-122B-A10B`, derived from the shared MoE definition with its model dimensions. Parallelism and optimizer choices remain recipe configuration.

`--streaming-model-checkpoint-save` selects a synchronous `torch_dist` model-only writer that avoids a full host staging copy. It requires `--no-save-optim`, persistent checkpoints, and a single full-model Megatron cell. It is not a replacement for resumable optimizer checkpoints and is incompatible with the SFT driver's full-state saves. `tests/manual/streaming_checkpoint_roundtrip.py` checks that the standard Megatron loader can read the result in a compatible Megatron environment.

Async training now evaluates restored weights before continuing when initial evaluation is enabled, and persists rollout state before publishing a model checkpoint.
