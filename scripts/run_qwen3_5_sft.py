"""Long-context Qwen3.5-122B SFT on twelve 8-GPU nodes.

Dataset/tokenizer paths and adapters are supplied by the private project.
Run under an existing Ray cluster (MILES_SCRIPT_EXTERNAL_RAY=1 on ABCI).
"""

import shlex
from dataclasses import dataclass

import typer

from miles.utils.external_utils import command_utils as U


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    run_id: str = U.create_run_id()
    model_dir: str = "/root/models"
    model_name: str = "Qwen3.5-122B-A10B"
    data_dir: str = "/root/datasets"
    validation_data: str = "/root/datasets/validation.parquet"
    checkpoint_dir: str = "/root/checkpoints/qwen35-sft"
    private_pythonpath: str = ""
    data_source_path: str = "miles.rollout.data_source.RolloutDataSource"
    rollout_function_path: str = "miles.rollout.sft_rollout.generate_rollout"
    megatron_path: str = "/root/Megatron-LM"
    num_gpus_per_node: int = 8
    actor_num_nodes: int = 12
    train_tp: int = 4
    train_cp: int = 2
    train_pp: int = 3
    train_ep: int = 32
    num_rollout: int = 4128
    keep_best: bool = True
    wandb_project: str = ""
    wandb_entity: str = ""
    extra_args: str = ""

    def __post_init__(self):
        world = self.actor_num_nodes * self.num_gpus_per_node
        model_parallel = self.train_tp * self.train_cp * self.train_pp
        if min(world, model_parallel, self.train_ep, self.num_rollout) <= 0 or world % model_parallel:
            raise ValueError("World size must be divisible by TP * CP * PP")
        if 128 % (world // model_parallel):
            raise ValueError("Global batch 128 must be divisible by data parallel size")
        if 48 % self.train_pp or 256 % self.train_ep or world % (self.train_ep * self.train_pp):
            raise ValueError("Layer/expert counts must be divisible by the chosen topology")


def _train_args(args):
    q = shlex.quote
    model = f"{args.model_dir}/{args.model_name}"
    flags = (
        f"--hf-checkpoint {q(model)} --ref-load {q(model)} "
        f"--load {q(args.checkpoint_dir)} --save {q(args.checkpoint_dir)} --save-interval 100 "
        f"--rollout-function-path {q(args.rollout_function_path)} --data-source-path {q(args.data_source_path)} "
        f"--prompt-data {q(args.data_dir)} --sft-validation-data {q(args.validation_data)} "
        "--input-key messages --loss-mask-type qwen3 --rollout-shuffle --rollout-seed 1 "
        f"--num-rollout {args.num_rollout} --num-epoch 6 --rollout-batch-size 128 "
        "--n-samples-per-prompt 1 --global-batch-size 128 --micro-batch-size 1 "
        "--loss-type sft_loss --calculate-per-token-loss --disable-compute-advantages-and-returns "
        "--debug-train-only --balance-data --sft-eval-interval 100 "
        f"--tensor-model-parallel-size {args.train_tp} --sequence-parallel "
        f"--pipeline-model-parallel-size {args.train_pp} --context-parallel-size {args.train_cp} "
        f"--expert-model-parallel-size {args.train_ep} --expert-tensor-parallel-size 1 "
        "--recompute-granularity full --recompute-method uniform --recompute-num-layers 1 "
        "--seq-length 65536 --log-probs-chunk-size 256 "
        "--optimizer adam --lr 2e-5 --lr-decay-style cosine --min-lr 2e-7 --lr-warmup-iters 123 "
        "--weight-decay 0.01 --adam-beta1 0.9 --adam-beta2 0.999 --clip-grad 1 --seed 42 "
        "--use-precision-aware-optimizer --main-params-dtype fp32 --exp-avg-dtype fp32 --exp-avg-sq-dtype fp32 "
        "--attention-dropout 0 --hidden-dropout 0 --accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 --attention-backend flash "
        f"--actor-num-nodes {args.actor_num_nodes} --actor-num-gpus-per-node {args.num_gpus_per_node} "
        f"--num-gpus-per-node {args.num_gpus_per_node} "
    )
    return flags + ("--sft-keep-best " if args.keep_best else "") + args.extra_args


def execute(args: ScriptArgs):
    wandb = shlex.split(U.get_default_wandb_args(__file__, run_id=args.run_id))
    if "--wandb-key" in wandb:
        index = wandb.index("--wandb-key")
        del wandb[index : index + 2]
    if wandb and args.wandb_project:
        wandb[wandb.index("--wandb-project") + 1] = args.wandb_project
    if wandb and args.wandb_entity:
        wandb += ["--wandb-entity", args.wandb_entity]
    args.create_backend().execute_train(
        train_args=_train_args(args) + " " + shlex.join(wandb),
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type="qwen3.5-122B-A10B",
        megatron_path=args.megatron_path,
        train_script="train_sft.py",
        job_lifetime="launcher",
        extra_env_vars={
            "PYTHONPATH": args.private_pythonpath,
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        },
    )


@U.dataclass_cli
def main(args: ScriptArgs):
    execute(args)


if __name__ == "__main__":
    typer.run(main)
