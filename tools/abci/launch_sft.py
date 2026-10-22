"""Launch a text-only SFT recipe inside an existing PBS allocation; never submits jobs."""

import argparse
import sys
from pathlib import Path

# This file also runs directly under the host Python, outside the container.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.abci import launch


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "repo",
        "image",
        "scratch",
        "model-dir",
        "model-name",
        "data-dir",
        "validation-data",
        "checkpoint-dir",
    ):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--private-pythonpath", default="")
    parser.add_argument("--container-root", default="")
    parser.add_argument("--credentials-file", default="")
    parser.add_argument("--data-source-path", default="miles.rollout.data_source.RolloutDataSource")
    parser.add_argument("--rollout-function-path", default="miles.rollout.sft_rollout.generate_rollout")
    parser.add_argument("--actor-nodes", type=int, default=12)
    parser.add_argument("--num-rollout", type=int, default=4128)
    parser.add_argument("--keep-best", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wandb-project", default="")
    parser.add_argument("--wandb-entity", default="")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--role", choices=["host", "head", "worker"], default="host")
    parser.add_argument("--head-ip", default="")
    parser.add_argument("--ray-port", type=int, default=6379)
    parser.add_argument("--dashboard-port", type=int, default=8265)
    parser.add_argument("--ray-object-store-gb", type=int, default=16)
    parser.add_argument("--extra-args", default="")
    parser.set_defaults(mode="sft")
    return parser


def _preflight(args):
    model = Path(args.model_dir) / args.model_name
    for path in (model / "config.json", model / "tokenizer_config.json", Path(args.validation_data)):
        if not path.is_file():
            raise FileNotFoundError(path)
    checkpoint = Path(args.checkpoint_dir)
    if checkpoint.resolve() == model.resolve():
        raise ValueError("Checkpoint output must not be the input model")
    tracker = checkpoint / "latest_checkpointed_iteration.txt"
    if tracker.exists():
        iteration = int(tracker.read_text())
        if not (checkpoint / f"iter_{iteration:07d}" / "sft_state.json").exists():
            raise ValueError("Refusing to reuse a non-SFT checkpoint directory")


def _execute_recipe(args):
    command = [sys.executable, "scripts/run_qwen3_5_sft.py"]
    for name in (
        "model_dir",
        "model_name",
        "data_dir",
        "validation_data",
        "checkpoint_dir",
        "private_pythonpath",
        "data_source_path",
        "rollout_function_path",
        "num_rollout",
        "wandb_project",
        "wandb_entity",
        "run_id",
        "extra_args",
    ):
        command += ["--" + name.replace("_", "-"), str(getattr(args, name))]
    command += ["--actor-num-nodes", str(args.actor_nodes), "--output-dir", str(Path(args.scratch) / "runs")]
    command += ["--keep-best" if args.keep_best else "--no-keep-best"]
    launch._run(command)


def main():
    args = _parser().parse_args()
    if args.role == "host":
        launch._host(
            args,
            entrypoint=Path(__file__).resolve(),
            expected_nodes=args.actor_nodes,
            required_inputs=[args.repo, args.image, args.model_dir, args.data_dir, args.validation_data],
        )
    else:
        launch._container_role(args, preflight=_preflight, execute_recipe=_execute_recipe)


if __name__ == "__main__":
    main()
