---
title: Supervised fine tuning
description: Train gold responses with token weighted validation and resumable latest and best checkpoints using the Megatron SFT driver.
---

`train_sft.py` runs synchronous supervised fine tuning on resident Megatron models. A rollout callable renders gold responses from the dataset; it does not generate responses through an inference engine. Dataset preparation, special tokens, hyperparameters, and cluster submission remain application configuration.

## Configuration

Use the normal model and optimizer arguments together with:

```text
--debug-train-only
--loss-type sft_loss
--disable-compute-advantages-and-returns
--calculate-per-token-loss
--rollout-function-path miles.rollout.sft_rollout.generate_rollout
--n-samples-per-prompt 1
--sft-validation-data /data/validation.parquet
--sft-eval-interval 100
--load /checkpoints/sft
--save /checkpoints/sft
--save-interval 100
--sft-keep-best
```

Set equal `--global-batch-size` and `--rollout-batch-size`, and choose a training horizon and batch layout appropriate for the corpus. Initialize fresh weights through `--hf-checkpoint`; the shared load/save directory is exclusively owned by this SFT run. Synchronous saves include optimizer, RNG, and data-cursor state. Neither optimizer/RNG omission nor Megatron's independent retention policy is supported by this driver.

The initial implementation supports one Megatron training cell, `torch_dist` checkpoints, fixed global batches, and THD/BSHD with standard context parallelism. Independent DP cells, training offload, separate evaluation GPUs, all-gather CP, dynamic global batches, and custom sample conversion/reward postprocessing are rejected. These restrictions apply to this driver, not to other Miles entrypoints.

## Rendering and validation

Training and validation share the configured dataset-field settings and rendering callable. A renderer returns one response per input in order, including tokens, response length, and a response-aligned loss mask. The default is `miles.rollout.sft_rollout.generate_rollout`. Application callables may implement other tokenization and masking rules. Stateful rollout classes are unsupported; async legacy callables require the legacy rollout-v1 runtime.

The standard training data source also supplies validation dataset construction by default. When using a custom training data source, explicitly set `--sft-validation-data-source-path` to a compatible validation reader. This reader receives the validation path and the same schema settings with shuffling disabled, and exposes its dataset through the normal data-source interface. It need not share training-only epoch or shard-loading behavior.

Validation renders one batch at a time. The final partial batch is padded with zero-loss examples so every real example contributes without changing the distributed schedule. Validation reports summed NLL divided by supervised-token count across DP/CP, counting each TP replica once. This differs from averaging per-example or per-batch losses when their supervised lengths differ.

Validation runs initially, at the requested interval, at epoch boundaries, and at the final update. It preserves model mode and training RNG. A validation event also triggers a save, so a newly best checkpoint can be retained. The cadence and loss definition should be accounted for when comparing with another trainer.

## Checkpoint completion and recovery

Each save stages SFT bookkeeping and persists the data cursor before synchronously saving the model and optimizer. Once their completion markers are verified, an atomic `sft_latest.json` manifest commits the update. Only then does the driver prune its older completed checkpoints. By default it retains the latest; `--sft-keep-best` also retains the best validated checkpoint.

On restart, the driver repairs Megatron's tracker from the committed manifest before workers load weights. An interrupted newer save cannot advance the model without its matching data cursor. Checkpoint roots from another trainer, or an earlier implementation without this manifest, require an explicit migration; do not point the driver at an unrelated checkpoint directory.

Keep the optimizer schedule consistent across interruption and restart. A resumed run infers its step from the committed checkpoint; manual step overrides and weights-only loading are rejected.

This capability has CPU contract tests. Distributed training, validation, and save/reload must be exercised with the intended model and topology before a production run. The tiled LM-head optimization is a separate feature; this driver continues to use the selected existing log-prob backend.
