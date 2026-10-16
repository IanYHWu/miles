---
title: ABCI with an external LLM judge
description: Run a Qwen3.5-9B CISPO baseline in a PBS allocation using Singularity.
---

The recipe is `scripts/run_qwen3_5_judge.py`. Its `ScriptArgs` owns the defaults.
The default is a 100-update Qwen3.5-9B regression baseline. It trains a single-turn policy with GRPO-normalized rewards and the CISPO objective,
using rollout log-probabilities, global token-mean reduction, and a separate SGLang
rollout pool. This is a text-only recipe; the vision tower and MTP head are not trained.

## Environment

Use a new image rather than modifying an existing training environment. The initial
candidate is the upstream image published as `dev-202610161956`, pinned by digest:

```bash
singularity pull miles.sif docker://radixark/miles@sha256:cb92aa30af0851927a87e8f53a32276a5d4aae6a2d194ef1dba7321e8f6a307a
```

The digest pins all image dependencies. Bind this Miles checkout into it so the new
CISPO and judge implementations take precedence. `tools/abci/preflight.py` checks
CUDA execution, dependency imports, all dataset rows, prompt lengths and judge
factories before training. CUDA/driver compatibility must pass on the allocated node.

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
- A judge JSON and the importable project package supplying its helpers. No prompt
  templates or dataset contents need to be copied into Miles.

Judge JSON schema (replace the example factory paths with your application's):

```json
{
  "endpoint_url": "https://api.openai.com/v1/chat/completions",
  "model": "YOUR_JUDGE_MODEL",
  "reasoning_effort": "medium",
  "api_key_env": "LLM_JUDGE_API_KEY",
  "max_concurrency": 64,
  "max_input_tokens": 32768,
  "max_output_tokens": 24576,
  "timeout_s": 900,
  "block_on_failure": true,
  "answer_parser_path": "my_project.readout.read_answer",
  "judges": {
    "rubric": {"factory": "my_project.judges.RubricJudge", "config": {}},
    "binary": {"factory": "my_project.judges.BinaryJudge", "config": {}}
  }
}
```

The answer parser returns an object with `answer` and `fault` attributes. A fault
receives zero without an HTTP call. Each judge factory takes a config dictionary;
its synchronous `build_prompt(problem, rubric, reference, answer)` returns text and
`parse_reply(text)` returns `reward` and `ok`. Miles never calls the helper's HTTP
methods. The answer token budget truncates only the answer, preserving the problem
and rubric, as in the baseline this adapter was built for.

Transient HTTP failures retry with capped backoff and honor cancellation. Permanent
4xx failures stop the run instead of silently assigning zero rewards. Parse failures
use the application parser's error score. Unknown score types, missing rubric/reference data,
or credentials fail explicitly.

## PBS smoke and production launch

`examples/platforms/abci/smoke.pbs` requests one node from project `gai51740`,
reservation `R9920261300`. Export its `MILES_*` inputs through `qsub -v`, including
`RTYPE=rt_HF`. Use an absolute scratch path under your own writable directory.

The smoke executes two updates, then restarts for one further update, with four trainer and four rollout GPUs, short
responses and alternating synthetic rewards. This guarantees nonzero learning
signal without paying a judge to grade truncated proofs. HTTP judge behavior has
separate unit tests. The smoke also checks trainer-to-rollout weights and saves a
checkpoint and verifies that restarting advances it. It is not a baseline result or a long-context capacity test.

Set `MILES_MODEL_NAME=Qwen3.5-122B-A10B` for the full-size MoE smoke. This uses
all eight GPUs with colocated training/rollout, TP2/EP8 training, TP8 inference,
rollout routing replay, and one short update. It saves model weights only; the
9B smoke covers optimizer checkpoint restart. Colocation requires `retract`
pauses and publishing weights after every update, unlike the disaggregated
baseline. To bound host memory, it reconstructs training weights from the optimizer's
master weights between phases, with the built-in bitwise reconstruction check enabled.
The 122B production topology is deliberately not selected by this recipe.

For the 100-update 9B baseline, `examples/platforms/abci/train.pbs` allocates two nodes and invokes `tools/abci/launch.py` with
`--mode train` explicitly; the default mode is read-only preflight. The wrapper uses
PBS to start the second node, a job-local Ray cluster, and separate PID namespaces.
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

## Porting differences to record

The framework/backend change is itself an experimental difference. SGLang replaces
vLLM, packed training replaces the previous padded path, and long-context memory
capacity still needs measurement. Fully async uses `in_place` pause with namespaced
prefix caching. Its integer `max_weight_staleness` is not equivalent to verl's
queue-capacity `staleness_threshold`; do not compare those numbers directly.

CISPO uses offsets from one: `--eps-clip 10 --eps-clip-high 4` caps the detached
importance ratio at five and effectively leaves the lower bound inactive. It does
not mean ratio bounds of ten and four. The log-ratio is clamped to [-20, 20] before
exponentiation. The PPO path remains the default for other recipes.

Resume from a Miles checkpoint using the recipe's `--load` argument. Optimizer-state
resume from a verl checkpoint is not supported. Before production, validate GPU
smoke, checkpoint restart, live judge scoring, and memory at the intended length.
