---
title: ABCI with an external LLM judge
description: Run Miles under PBS on ABCI with Singularity, a job-local Ray cluster and an external LLM judge.
---

The recipe is `scripts/run_qwen3_5_judge.py`. Its `ScriptArgs` owns the defaults.
Its default is a 100-update Qwen3.5-9B run. It trains a single-turn policy with GRPO-normalized rewards and the CISPO objective,
using rollout log-probabilities, global token-mean reduction, and a separate SGLang
rollout pool. This is a text-only recipe; the vision tower and MTP head are not trained.

## Environment

Use a new image rather than modifying an existing training environment. The tested
image is the upstream image published as `dev-202610161956`, pinned by digest:

```bash
singularity pull miles.sif docker://radixark/miles@sha256:cb92aa30af0851927a87e8f53a32276a5d4aae6a2d194ef1dba7321e8f6a307a
```

The digest pins all image dependencies. Bind this Miles checkout into it so the new
CISPO and judge implementations take precedence. `tools/abci/preflight.py` checks
CUDA execution, dependency imports, prompt lengths, and that the judge adapter accepts
every dataset row before training. CUDA/driver compatibility must pass on the allocated node.

Use a gzip-compressed SIF on ABCI. The tested site runtime rejects Zstandard
SquashFS, while the kernel has LZ4 support disabled. GPU execution from an unpacked
sandbox on Lustre also fails because the NVIDIA runtime needs a writable overlay.

The Docker image also keeps editable dependency sources under `/root`, which is
not readable to a normal cluster user after conversion. Extract its `root` tree
into owned scratch storage with `unsquashfs -o OFFSET -d SUPPORT_DIR miles.sif root`
(find the filesystem offset with `singularity sif list` / `singularity sif info`).
Set `MILES_CONTAINER_ROOT=SUPPORT_DIR/root`; the launcher binds this tree read-only
at `/root`. It must come from the same pinned image, so Megatron and the installed
packages stay matched.

## Private inputs

Stage these outside the public repository:

- `MODEL_DIR/Qwen3.5-9B`: complete Hugging Face checkpoint and tokenizer.
- `DATA_DIR/train.parquet` and `val.parquet`: existing materialized datasets, or
  symlinks to them. Required fields are `prompt` (chat messages), `reward_model`
  (`ground_truth` rubric), and `extra_info` (`raw_problem`, `score_type`,
  `reference_answer`). Prompt lengths must fit the recipe's budget.
- A judge JSON for `miles.rollout.llm_judge.reward` and the importable project
  package supplying its adapter (`MILES_PRIVATE_PYTHONPATH`). No prompt templates or
  dataset contents need to be copied into Miles.

The judge contract is described in [Judged training and CISPO](/user-guide/judged-training):
the JSON names an `adapter_factory`, whose adapter interprets the dataset fields above,
builds the judge messages and parses replies into a reward dictionary. The recipe trains
on its `score` key. For example:

```json
{
  "endpoint_url": "https://api.openai.com/v1/chat/completions",
  "model": "YOUR_JUDGE_MODEL",
  "api_key_env": "LLM_JUDGE_API_KEY",
  "adapter_factory": "my_project.judge.make_adapter",
  "adapter_config": {"max_input_tokens": 32768},
  "max_concurrency": 64,
  "timeout_s": 900,
  "block_on_failure": true,
  "generation_kwargs": {"max_completion_tokens": 24576, "temperature": 1.0, "top_p": 1.0, "reasoning_effort": "medium"}
}
```

Answer extraction, truncating long answers to a grader budget, and the reward for
an answer the application rejects are adapter decisions. With `block_on_failure`
transient HTTP failures retry indefinitely with capped backoff, so exhaustion stops
the run rather than assigning zero rewards; without it the attempt limit is finite.
Errors include bounded provider codes and request IDs, never raw messages.

## PBS smoke and production launch

`examples/platforms/abci/smoke.pbs` requests one node. Submit it with your own
group and queue or reservation (`qsub -P GROUP -q QUEUE`), and export its `MILES_*`
inputs through `qsub -v`, including `RTYPE=rt_HF`: a command-line `-v` replaces the
template's own. Use an absolute scratch path under your own writable directory.

The smoke executes two updates, then restarts for one further update, with four trainer and four rollout GPUs, short
responses and alternating synthetic rewards. This guarantees nonzero learning
signal without paying a judge to grade truncated proofs. HTTP judge behavior has
separate unit tests. The smoke also checks trainer-to-rollout weights and saves a
checkpoint and verifies that restarting advances it. It is not a training result or a long-context capacity test.
Before synthetic training, preflight sends one live request through the judge
transport and scores one training row through the adapter, requiring a reward with
a `score`. This verifies the endpoint, credential and adapter end to end without
treating the synthetic training rewards as a judge evaluation; whether the score is
correct for that row is the adapter's own test.

Set `MILES_SMOKE_MODE=async-smoke` on the same one-node PBS script for a 9B
async stress test. It uses separate trainer and rollout GPUs, native requests
of varying lengths, repeated weight publications, and synthetic rewards. The
wrapper fails unless mixed-version trajectories reach training, all gradient
norms are finite and positive, published weights change, and the final checkpoint
exists. Token-version and log-probability alignment are checked on completed
samples. Evidence lives in `async-evidence.jsonl` and `async-result.json` in the
run directory. This test does not repeat the basic smoke's checkpoint restart.
Its CI sample-ownership checker allows eight steps for an issued sample to
resolve: the default two-step grace period can misclassify legitimate long or
buffered trajectories as lost. This allowance does not relax the policy's
staleness filter or the check against duplicate training consumption.

Production leaves the token-age cutoff unset and bounds generation through concurrency
and the completed buffer. The async stress smoke explicitly uses a limit of six
optimizer versions to test stale-group rejection; it is not a production recommendation.
At the two-update publication cadence, six versions cover three publication gaps.

Production keeps only the latest full model/optimizer checkpoint. Megatron finalizes
the new save before deleting the previous one; its permanent-retention milestone is
set beyond the run horizon. A replacement can temporarily coexist with the previous save.

Set `MILES_MODEL_NAME=Qwen3.5-122B-A10B` for the full-size MoE smoke. This uses
all eight GPUs with colocated training/rollout, TP4/EP8 training, TP8 inference,
rollout routing replay, and one short update. It saves model weights only; the
9B smoke covers optimizer checkpoint restart. Colocation requires `retract`
pauses and publishing weights after every update, unlike the disaggregated
configuration. CPU Adam's full gradient copy exceeds the single node's host-memory
budget. This smoke instead executes precision-aware GPU Adam with chunked CPU
state offload, FP32 moments, and exact master weights stored using TE's parameter
remainders. This avoids a full CPU gradient copy and limits GPU moment residency
to small windows. The 9B recipe retains CPU optimizer offload. GPU capacity must
include the first optimizer step, not just loading and backward.
During the handoff to inference, GPU-resident trainer state is backed to the
run's `offload` directory on disk, preventing another host-memory copy. This is
separate from streaming the entire optimizer: the canonical optimizer moments
remain in RAM. Allow at least 600 GiB of free quota for the temporary backing
files and model checkpoint. Use a real disk filesystem; ABCI's `/tmp` is tmpfs.
The model-only smoke checkpoint uses `--streaming-model-checkpoint-save` to write
tensors incrementally through PyTorch's synchronous writer. It preserves Megatron's
distributed checkpoint keys and shard metadata while avoiding a full additional
CPU staging copy. This option requires synchronous, persistent, model-only saves;
the 9B optimizer checkpoint path is unchanged.
For dedicated 122B training, set `MILES_MODEL_NAME=Qwen3.5-122B-A10B`,
`MILES_ACTOR_NODES=4`, `MILES_ROLLOUT_NODES=4`, `MILES_TRAIN_TP=4`,
`MILES_TRAIN_CP=2`, `MILES_TRAIN_EP=32`, and `MILES_ROLLOUT_TP=8`,
then request eight nodes. `MILES_OPTIMIZER_CPU_OFFLOAD=0` selects GPU Adam
with FP32 moments; production saves include optimizer state. This configuration
uses rollout routing replay and disaggregated inference with `in_place` pauses.
Long-sequence peak memory must still be measured on the allocated GPUs.

`MILES_CHECKPOINT_DIR` may select a dedicated directory on group storage. The
wrapper binds that directory writable while keeping the rest of `/groups`
read-only. `MILES_RAY_OBJECT_STORE_GB` sizes each node's object store; routed
expert traces at long context are substantially larger than dense-model samples.
For a fresh run, `MILES_VERIFY_RESUME_AFTER_UPDATES=5` stops after the first
five-update checkpoint, checks its tracker, and relaunches from it with validation
before continuing to the unchanged total target. It fails closed if the first
phase fails or the expected checkpoint is missing.

The production PBS template requests 64 hours. For a continuation, set `MILES_LOAD`
and `MILES_CHECKPOINT_DIR` to the same existing checkpoint root. This restores
training and rollout state and lets retention delete the previous checkpoint
after its replacement succeeds, without copying model weights. `MILES_NUM_ROLLOUT`
is the total update target, including restored updates.

For the default 100-update 9B run, `examples/platforms/abci/train.pbs` allocates two nodes and invokes `tools/abci/launch.py` with
`--mode train` explicitly; the default mode is read-only preflight. Set `MILES_ROLLOUT_NODES=3` and `MILES_NUM_ROLLOUT=120`, and override PBS
with `-l select=4`, for one trainer node plus three rollout nodes. The wrapper uses
PBS to start each additional node, a job-local Ray cluster, and separate PID namespaces.
It does not run user-wide process cleanup or alter another allocation. Credentials
are loaded within each container, outside command-line arguments and Ray runtime
environment logs. `--credentials-file` optionally reads the local colon-delimited
`OpenAI API` and `Wandb` entries. Never commit that file.

The Python recipe also supports an existing Ray cluster with
`MILES_SCRIPT_EXTERNAL_RAY=1`. Set `PYTHONPATH` to the private helper package and
provide the judge credential in every node's environment before starting Ray.
W&B is enabled when `WANDB_API_KEY` is present, except in smoke mode. Set
`--wandb-project` and `--wandb-entity` on the Python recipe to select a destination.
The recipe retains the standard naming helper while removing its credential from
argv; authentication uses the inherited environment.

## Algorithm and resume notes

Fully async uses `in_place` pause with namespaced prefix caching. Its integer
`max_weight_staleness` counts optimizer versions; it is not a queue capacity.

CISPO uses offsets from one: `--eps-clip 10 --eps-clip-high 4` caps the detached
importance ratio at five and effectively leaves the lower bound inactive. It does
not mean ratio bounds of ten and four. The log-ratio is clamped to [-20, 20] before
exponentiation. The PPO path remains the default for other recipes.

Resume from a Miles checkpoint using the recipe's `--load` argument. Before production, validate GPU
smoke, checkpoint restart, live judge scoring, and memory at the intended length.

For a validated continuation, set `MILES_EVAL_BEFORE_TRAIN=1`: the async driver
evaluates the restored policy before requesting any training batch. Set
`MILES_SAVE_INTERVAL=5` (the default) to save every five optimizer updates.
Rollout state is saved before publishing each model checkpoint.

The optional judge JSON `error_log_dir` stores provider error bodies in mode-600
files for local diagnosis. Treat that directory as private: provider messages may
echo prompt text. Public logs contain only error codes and request identifiers.
