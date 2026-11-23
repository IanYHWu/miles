---
title: Driving a long-lived trainer
description: Trainer-group calls for custom driver scripts that keep one Megatron trainer alive across phases and runs.
---
`train.py` runs a fixed loop: generate, train, sync weights, save. An application with its own schedule — for example one that alternates RL steps with supervised steps on the same model, or that resumes a trainer across jobs — can instead run its own driver script in place of `train.py`. It builds the trainer with `create_training_models` and calls methods on the returned trainer group, as `train.py` does with `train`, `save_model` and `export_hf`.

The calls below act on one Megatron instance. They require exactly one trainer cell (no `--indep-dp`, no fault-tolerant spare cells), and the trainer group refuses them otherwise. Each returns one entry per trainer rank; collective ones answer on every rank. All of them need the Megatron backend with the distributed optimizer (`--use-distributed-optimizer`).

To push weights into vLLM servers started outside miles, see [vLLM Weight Push](/advanced/vllm-weight-push).

## Several optimizer histories on one model

Two objectives trained in alternation on the same weights usually should not share Adam moments. `swap_optimizer_state(stash_as, load)` keeps one history per name: the active one (moments and step counters) is moved to host memory as `stash_as`, and `load` becomes active, restored in place or fresh (zero moments, step 0) the first time. The fp32 main params are shared by every history and are not touched.

```python
await actor.swap_optimizer_state("rl", "sft")  # RL history to host memory, SFT history active
...  # SFT steps
await actor.swap_optimizer_state("sft", "rl")
```

A Megatron checkpoint (`--save`) holds only the active history. `save_stashed_optimizer_states(directory)` writes the stashed ones next to it (one file per rank), and `load_stashed_optimizer_states(directory)` reads them back after a restart from that checkpoint; a directory without them loads nothing. Histories are refused when the optimizer class has no known history schema (Adam and the Muon variants have one), and a restore is checked against the optimizer's layout before anything is written.
