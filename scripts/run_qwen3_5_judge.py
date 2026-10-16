"""Qwen3.5 single-turn CISPO with an external LLM judge.

Requires local HF weights, parquet prompts, and an application-owned judge JSON.
Args:
    model_dir: Directory containing the selected Qwen3.5 checkpoint.
    data_dir: Directory containing train.parquet and val.parquet.
    judge_config: Judge JSON; factories and answer parser must be importable.
    smoke: Short synthetic-reward validation; 122B uses eight colocated GPUs.

Example:
    python scripts/run_qwen3_5_judge.py --model-dir /models --data-dir /data \
        --judge-config /config/judge.json --smoke
"""

import shlex
from dataclasses import dataclass, field
from typing import Literal

import typer

from miles.utils.external_utils import command_utils as U


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    run_id: str = field(default_factory=lambda: U.create_run_id())
    model_dir: str = "/root/models"
    model_name: Literal["Qwen3.5-9B", "Qwen3.5-122B-A10B"] = "Qwen3.5-9B"
    data_dir: str = "/root/datasets"
    judge_config: str = "/root/datasets/judge.json"
    megatron_path: str = "/root/Megatron-LM"
    private_pythonpath: str = ""
    num_gpus_per_node: int = 8
    actor_num_nodes: int = 1
    actor_gpus: int = 8
    rollout_gpus: int = 8
    train_tp: int = 4
    rollout_tp: int = 1
    num_rollout: int = 100
    rollout_batch_size: int = 32
    samples_per_prompt: int = 16
    response_length: int = 92160
    prompt_length: int = 2048
    context_length: int = 94464
    lr: float = 5e-7
    clip_low: float = 10.0
    clip_high: float = 4.0
    update_interval: int = 2
    max_weight_staleness: int = 1
    concurrency: int = 1024
    eval_interval: int = 30
    save_interval: int = 20
    smoke: bool = False
    load: str = ""
    extra_args: str = ""
    wandb_project: str = ""
    wandb_entity: str = ""

    def __post_init__(self):
        if self.model_name == "Qwen3.5-122B-A10B" and not self.smoke:
            raise ValueError("The 122B production topology must be sized separately; use --smoke for validation")
        if self.smoke:
            self.actor_num_nodes, self.actor_gpus, self.rollout_gpus = 1, 4, 4
            self.num_rollout, self.rollout_batch_size, self.samples_per_prompt = 2, 2, 2
            self.response_length, self.context_length, self.concurrency = 128, 4096, 8
            if self.model_name == "Qwen3.5-122B-A10B":
                # Colocation fits the full 122B model on the one-node test budget.
                self.actor_gpus, self.rollout_gpus = 8, 8
                self.train_tp, self.rollout_tp = 4, 8
                self.samples_per_prompt = 4
                self.update_interval, self.num_rollout = 1, 1
        if self.actor_gpus % self.train_tp or self.rollout_gpus % self.rollout_tp:
            raise ValueError("GPU counts must be divisible by their tensor parallel sizes")


def _train_args(args):
    q = shlex.quote
    model = f"{args.model_dir}/{args.model_name}"
    colocate = args.model_name == "Qwen3.5-122B-A10B"
    checkpoint = (
        f"--hf-checkpoint {q(model)} --ref-load {q(model)} --load {q(args.load or model)} "
        f"--save {q(f'{args.output_dir}/{args.run_id}/checkpoints')} "
        f"--save-interval {2 if args.smoke else args.save_interval} "
    )
    rollout = (
        f"--prompt-data {q(args.data_dir + '/train.parquet')} --input-key prompt "
        "--label-key reward_model --metadata-key extra_info --apply-chat-template --rollout-shuffle "
        "--custom-rm-path miles.rollout.llm_judge.reward --reward-key score "
        f"--llm-judge-config {q(args.judge_config)} --num-rollout {args.num_rollout} "
        f"--rollout-batch-size {args.rollout_batch_size} --n-samples-per-prompt {args.samples_per_prompt} "
        f"--global-batch-size {args.rollout_batch_size * args.samples_per_prompt} "
        f"--rollout-max-response-len {args.response_length} --rollout-max-prompt-len {args.prompt_length} "
        "--rollout-temperature 1 --rollout-top-p 1 --rollout-top-k -1 --balance-data "
        f"--fully-async --pause-generation-mode {'retract' if colocate else 'in_place'} "
        f"--update-weights-interval {args.update_interval} --max-weight-staleness {args.max_weight_staleness} "
        f"--async-max-concurrent-samples {args.concurrency} "
    )
    algorithm = (
        "--advantage-estimator grpo --policy-loss-type cispo --use-rollout-logprobs "
        "--skip-actor-forward-only --calculate-per-token-loss --observe-training-entropy "
        f"--eps-clip {args.clip_low} --eps-clip-high {args.clip_high} --entropy-coef 0 --kl-coef 0 "
    )
    optimizer = (
        f"--optimizer adam --lr {args.lr} --lr-decay-style constant --weight-decay 0.01 "
        "--adam-beta1 0.9 --adam-beta2 0.999 --clip-grad 1 "
    )
    performance = (
        f"--seed 42 --tensor-model-parallel-size {args.train_tp} --sequence-parallel "
        "--pipeline-model-parallel-size 1 --context-parallel-size 1 --micro-batch-size 1 "
        "--recompute-granularity full --recompute-method uniform --recompute-num-layers 1 "
        "--log-probs-chunk-size 256 --attention-dropout 0 --hidden-dropout 0 "
        "--accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32 --attention-backend flash "
    )
    engines = (
        f"--num-gpus-per-node {args.num_gpus_per_node} --actor-num-nodes {args.actor_num_nodes} "
        f"--actor-num-gpus-per-node {args.actor_gpus} "
        f"--rollout-num-gpus-per-engine {args.rollout_tp} --sglang-context-length {args.context_length} "
        "--sglang-mem-fraction-static 0.85 --sglang-max-running-requests 64 --sglang-chunked-prefill-size 8192 "
    )
    if colocate:
        engines += "--colocate --use-rollout-routing-replay --no-save-optim --sglang-disable-cuda-graph "
        engines += "--streaming-model-checkpoint-save "
        performance += "--expert-model-parallel-size 8 --expert-tensor-parallel-size 1 "
        # CPU Adam retains a second full gradient copy. Stage optimizer state in
        # small GPU windows instead, retaining FP32 moments and TE's exact masters.
        optimizer += (
            "--use-precision-aware-optimizer --chunked-optimizer-state-offload "
            "--optimizer-state-offload-fraction 1.0 --optimizer-state-offload-chunk-size-mb 256 "
            "--main-params-dtype fp32 --exp-avg-dtype fp32 --exp-avg-sq-dtype fp32 "
        )
    else:
        optimizer += "--optimizer-cpu-offload --overlap-cpu-optimizer-d2h-h2d --use-precision-aware-optimizer "
        engines += f"--rollout-num-gpus {args.rollout_gpus} --namespaced-radix-cache "
    evaluation = (
        ""
        if args.smoke
        else (
            f"--eval-prompt-data validation {q(args.data_dir + '/val.parquet')} --eval-interval {args.eval_interval} "
            f"--n-samples-per-eval-prompt 3 --eval-max-response-len {args.response_length} "
            "--eval-temperature 1 --eval-top-p 1 --eval-top-k -1 --skip-eval-before-train "
        )
    )
    return checkpoint + rollout + algorithm + optimizer + performance + engines + evaluation + args.extra_args


def execute(args: ScriptArgs):
    backend = args.create_backend()
    # Preserve standard W&B naming, but keep its credential out of job argv/logs.
    # Each Ray node must inherit WANDB_API_KEY before it starts.
    wandb = shlex.split(U.get_default_wandb_args(__file__, run_id=args.run_id)) if not args.smoke else []
    if "--wandb-key" in wandb:
        i = wandb.index("--wandb-key")
        del wandb[i : i + 2]
    if wandb and args.wandb_project:
        wandb[wandb.index("--wandb-project") + 1] = args.wandb_project
    if wandb and args.wandb_entity:
        wandb += ["--wandb-entity", args.wandb_entity]
    backend.execute_train(
        train_args=_train_args(args) + " " + shlex.join(wandb),
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=args.model_name.replace("Qwen", "qwen"),
        train_script="train_async.py",
        megatron_path=args.megatron_path,
        job_lifetime="launcher",
        extra_env_vars={
            "MR_SKIP_VERL_REGISTRATION": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "PYTHONPATH": args.private_pythonpath,
        },
    )


@U.dataclass_cli
def main(args: ScriptArgs):
    execute(args)


if __name__ == "__main__":
    typer.run(main)
