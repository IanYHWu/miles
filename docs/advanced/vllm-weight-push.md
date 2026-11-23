---
title: vLLM Weight Push
description: Push trainer weights over NCCL into vLLM servers that run outside miles.
---
miles' regular weight sync (`--update-weight-transfer-mode`) updates the SGLang engines that miles starts and owns. When rollouts come from vLLM servers started elsewhere — for example by an application's own agent loop on other nodes — a custom training driver can push the trainer's weights into them directly, using vLLM's RLHF weight-transfer API. The miles image does not need vLLM for this; the trainer speaks the NCCL side of the protocol itself.

## Requirements

- Each vLLM server runs with `--weight-transfer-config '{"backend": "nccl"}'` and `VLLM_SERVER_DEV_MODE=1`, which exposes `/init_weight_transfer_engine`, `/start_weight_update`, `/update_weights`, `/finish_weight_update`, `/pause`, `/resume`, `/weight_info`, `/update_weight_version` and `/reset_prefix_cache`.
- A copy of the serving image's `libnccl.so`, readable from the trainer nodes. NCCL refuses communicators whose ranks run different versions, and the trainer's torch usually links another one; the trainer loads this copy with ctypes.
- `--save-hf` (any path): the push streams the same Megatron→HF weight iterator the HF export uses, so the tensor names are exactly the ones vLLM loads from disk. Nothing is written there unless the driver saves or exports.
- One trainer cell (no `--indep-dp`, no fault-tolerant spare cells).
- `--distributed-timeout-minutes` longer than a push: while global rank 0 broadcasts, the other ranks wait in the iterator's gathers.

## Driver API

A custom driver script, run in place of `train.py`, calls two methods on the trainer group returned by `create_training_models` (see [Driving a long-lived trainer](/user-guide/trainer-control)):

```python
connected = await actor.vllm_connect(
    [{"url": "http://node3:8000", "world_size": 8}, ...],  # one entry per server
    "/shared/libnccl.so.2",
    http_timeout=300.0,
    connect_timeout=600.0,
    push_timeout=900.0,
)
pushed = await actor.vllm_push("17")  # publishes weight version "17"
```

Both return one entry per trainer rank; global rank 0 answers with a dict, the other ranks with `None`. Failures are returned as `{"ok": False, "error": ...}` instead of raised, because a raise inside a trainer call kills the trainer cell: a driver can re-read its server list, reconnect to the servers that still answer, and push again.

`vllm_connect` also takes `extra_ranks` and `test_batch_delay`, fault-injection knobs for validating recovery (a rendezvous that must time out; a push slow enough to kill a server during it). Leave them at their defaults otherwise.

## How it works

The implementation lives in `miles/backends/training_utils/weight_update/vllm_nccl/`.

1. **Connect**: global trainer rank 0 opens one NCCL communicator per server, all rendezvousing concurrently by NCCL unique id. The trainer is rank 0 of each communicator and the server's GPUs follow. A server that rejects the init is dropped; one whose rendezvous does not finish within `connect_timeout` is reported in `wedged` (stuck in its own `ncclCommInitRank`; do not use it again).
2. **Push**: every server is paused with `mode=keep` (in-flight requests freeze and continue under the new weights), then for each HF weight batch rank 0 posts its names, dtypes and shapes to every server's `/update_weights` and broadcasts the tensors on each server's communicator. Afterwards `/finish_weight_update` publishes the version, the prefix cache is reset (pre-empted requests re-prefill, so no KV computed with the old weights is reused), and the servers resume. Every server must then report the new version.
3. **Failures**: with one communicator per server, a server that dies mid-push only breaks its own link. It is aborted and dropped, its version is set to `poisoned` so it never serves partial weights under a version, and the push completes on the others. A watchdog aborts every link if a push exceeds `push_timeout`.

Weights the trainer does not hold (for example a frozen MTP draft or vision tower) are not sent; vLLM keeps the loaded tensors of modules that receive nothing.
