"""Bootstrap a job-local Ray cluster in Singularity under an existing PBS allocation.

No job is submitted by this command. Use --mode smoke for short synthetic-reward
updates; --mode train must be selected explicitly for real judge-scored training.
Runtime outputs belong under --scratch; --checkpoint-dir may use separate group storage.
Credentials never appear in argv.
"""

import argparse
import os
import shlex
import socket
import subprocess
import sys
import time
from pathlib import Path


def _parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["preflight", "smoke", "async-smoke", "train"], default="preflight")
    for name in ("image", "repo", "scratch", "model-dir", "data-dir", "judge-config"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--private-pythonpath", default="")
    p.add_argument("--container-root", default="", help="Readable extracted /root from the same pinned image")
    p.add_argument("--model-name", choices=["Qwen3.5-9B", "Qwen3.5-122B-A10B"], default="Qwen3.5-9B")
    p.add_argument("--credentials-file", default="")
    p.add_argument("--load", default="")
    p.add_argument("--checkpoint-dir", default="")
    p.add_argument("--save-interval", type=int, default=5)
    p.add_argument("--eval-before-train", action="store_true")
    p.add_argument("--actor-nodes", type=int, default=1)
    p.add_argument("--rollout-nodes", type=int, default=1)
    p.add_argument("--train-tp", type=int, default=4)
    p.add_argument("--train-cp", type=int, default=1)
    p.add_argument("--train-ep", type=int, default=1)
    p.add_argument("--rollout-tp", type=int, default=1)
    p.add_argument("--optimizer-cpu-offload", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--verify-resume-after-updates", type=int, default=0)
    p.add_argument("--num-rollout", type=int, default=100)
    p.add_argument("--wandb-project", default="")
    p.add_argument("--wandb-entity", default="")
    p.add_argument("--role", choices=["host", "head", "worker"], default="host")
    p.add_argument("--head-ip", default="")
    p.add_argument("--ray-port", type=int, default=6379)
    p.add_argument("--dashboard-port", type=int, default=8265)
    p.add_argument("--ray-object-store-gb", type=int, default=16)
    return p


def _run(argv, **kwargs):
    subprocess.run(argv, check=True, **kwargs)


def _run_head(command, workers):
    head = subprocess.Popen(command)
    try:
        while head.poll() is None:
            for worker in workers:
                if worker.poll() is not None:
                    raise RuntimeError(f"PBS worker exited early with status {worker.returncode}")
            time.sleep(1)
        if head.returncode:
            raise subprocess.CalledProcessError(head.returncode, command)
    finally:
        if head.poll() is None:
            head.terminate()
            try:
                head.wait(timeout=30)
            except subprocess.TimeoutExpired:
                head.kill()
                head.wait()


def _credentials(path):
    if not path:
        return
    labels = {"wandb": "WANDB_API_KEY", "openai api": "LLM_JUDGE_API_KEY"}
    for line in Path(path).read_text().splitlines():
        label, sep, value = line.partition(":")
        if sep and label.strip().lower() in labels:
            os.environ[labels[label.strip().lower()]] = value.strip()


def _preflight(args):
    command = [
        sys.executable,
        "tools/abci/preflight.py",
        "--model-dir",
        args.model_dir,
        "--model-name",
        args.model_name,
        "--data-dir",
        args.data_dir,
        "--judge-config",
        args.judge_config,
    ]
    if args.mode in ("smoke", "async-smoke"):
        command += ["--probe-judge"]
    _run(command)


def _execute_recipe(args):
    run_id = "pbs-" + os.environ["PBS_JOBID"].split(".")[0] + "-" + args.model_name
    run_dir = Path(args.scratch) / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    argv = [
        sys.executable,
        "scripts/run_qwen3_5_judge.py",
        "--model-dir",
        args.model_dir,
        "--model-name",
        args.model_name,
        "--data-dir",
        args.data_dir,
        "--judge-config",
        args.judge_config,
        "--private-pythonpath",
        args.private_pythonpath,
        "--output-dir",
        str(run_dir.parent),
        "--run-id",
        run_id,
    ]
    if args.mode not in ("smoke", "async-smoke"):
        argv += [
            "--load",
            args.load,
            "--checkpoint-dir",
            args.checkpoint_dir,
            "--save-interval",
            str(args.save_interval),
            "--wandb-project",
            args.wandb_project,
            "--wandb-entity",
            args.wandb_entity,
            "--actor-num-nodes",
            str(args.actor_nodes),
            "--train-tp",
            str(args.train_tp),
            "--train-cp",
            str(args.train_cp),
            "--train-ep",
            str(args.train_ep),
            "--rollout-tp",
            str(args.rollout_tp),
            "--optimizer-cpu-offload" if args.optimizer_cpu_offload else "--no-optimizer-cpu-offload",
            "--rollout-gpus",
            str(8 * args.rollout_nodes),
            "--num-rollout",
            str(args.num_rollout),
        ]
        if args.eval_before_train:
            argv.append("--no-skip-eval-before-train")
        if args.verify_resume_after_updates:
            _verify_training_resume(args, argv, run_dir)
        else:
            _run(argv)
        return
    import torch

    # Each allocation has its own evidence: stale files cannot pass a new smoke.
    checkpoint = run_dir / "checkpoints"
    if (checkpoint / "latest_checkpointed_iteration.txt").exists():
        raise RuntimeError(f"Smoke output already exists: {run_dir}")
    extra = (
        "--custom-rm-path tests.manual.judged_baseline_fixture.reward "
        # Gradient artifacts require CI mode. Its zero-KL assertion assumes
        # recomputed actor logprobs, unlike this rollout-logprob CISPO recipe.
        # Control the HF equality check explicitly so resume can turn it off.
        "--ci-test --ci-disable-kl-checker --ci-disable-weight-update-checker "
        "--check-weight-update-equal --check-weight-update-skip-list visual mtp "
        f"--ci-save-grad-norm {shlex.quote(str(run_dir / 'grad-{rollout_id}.pt'))} "
        "--log-inference-engine-weight-checksums"
    )
    if args.mode == "async-smoke":
        extra += (
            # The CI default (two steps) declares legitimate long/in-flight or
            # buffered groups lost before the async consumer can resolve them.
            " --sample-ownership-grace-steps 8 --custom-generate-function-path tests.manual.async_baseline_fixture.generate --custom-rollout-log-function-path tests.manual.async_baseline_fixture.log_rollout_data --custom-megatron-before-train-step-hook-path tests.manual.async_baseline_fixture.before_train"
        )
        _run(argv + ["--async-smoke", "--extra-args", extra])
        # A fresh interpreter picks up the repository PYTHONPATH set by the
        # container bootstrap; changing os.environ does not alter sys.path here.
        _run([sys.executable, "-m", "tests.manual.async_baseline_fixture", str(run_dir)])
        print("GPU async smoke: 9B, mixed-version training, repeated weight sync, finite gradients", flush=True)
        return
    restart = args.model_name == "Qwen3.5-9B"
    # Keep the scheduler horizon identical across interruption and restart.
    initial_extra = extra + (" --num-rollout 3 --debug-exit-after-rollout 2" if restart else "")
    _run(argv + ["--smoke", "--extra-args", initial_extra])
    tracker = checkpoint / "latest_checkpointed_iteration.txt"
    if not tracker.exists():
        raise RuntimeError("Smoke did not save a resumable checkpoint")
    iteration = int(tracker.read_text().strip())
    if restart:
        # Relaunch the complete stack and require a fresh update after restoring.
        # The initial equality check compares against the original HF weights,
        # so it is only valid before training, not after loading a checkpoint.
        resume_extra = extra.replace("--check-weight-update-equal ", "")
        _run(argv + ["--smoke", "--load", str(checkpoint), "--extra-args", resume_extra + " --num-rollout 3"])
    for step in range(3 if restart else 1):
        norm = float(torch.load(run_dir / f"grad-{step}.pt", weights_only=False))
        if not 0 < norm < float("inf"):
            raise RuntimeError(f"Smoke step {step} has invalid gradient norm: {norm}")
    if restart and int(tracker.read_text().strip()) <= iteration:
        raise RuntimeError("Resumed smoke did not advance the checkpoint")
    print(
        f"GPU smoke: {args.model_name}, finite nonzero gradients, weight sync, checkpoint saved; restart={restart}",
        flush=True,
    )


def _verify_training_resume(args, argv, run_dir):
    """Stop after a real save, then restore the full training stack before continuing."""
    updates = args.verify_resume_after_updates
    if args.load or updates <= 0 or updates >= args.num_rollout or updates % args.save_interval:
        raise ValueError("Resume verification needs a fresh run and a save-aligned stop before the total target")
    checkpoint = Path(args.checkpoint_dir) if args.checkpoint_dir else run_dir / "checkpoints"
    tracker = checkpoint / "latest_checkpointed_iteration.txt"
    if tracker.exists():
        raise RuntimeError(f"Refusing to overwrite existing training checkpoint: {checkpoint}")
    _run(argv + ["--extra-args", f"--debug-exit-after-rollout {updates}"])
    if not tracker.is_file() or int(tracker.read_text()) != updates - 1:
        raise RuntimeError("Initial training phase did not publish the expected checkpoint")
    resumed = list(argv)
    resumed[resumed.index("--load") + 1] = str(checkpoint)
    if "--no-skip-eval-before-train" not in resumed:
        resumed.append("--no-skip-eval-before-train")
    print(f"Restoring full model, optimizer and rollout checkpoint after {updates} updates", flush=True)
    _run(resumed)


def _container_role(args):
    _credentials(args.credentials_file)
    os.environ.update(
        {
            "MR_SKIP_VERL_REGISTRATION": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "PYTHONPATH": ":".join(filter(None, [args.repo, "/root/Megatron-LM", args.private_pythonpath])),
            "MILES_SCRIPT_EXTERNAL_RAY": "1",
            "MASTER_ADDR": args.head_ip,
            "RAY_ADDRESS": f"http://{args.head_ip}:{args.dashboard_port}",
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
            "NCCL_NVLS_ENABLE": "0",
            "WANDB_DIR": args.scratch + "/wandb",
            "WANDB_CACHE_DIR": args.scratch + "/wandb-cache",
            "WANDB_CONFIG_DIR": args.scratch + "/wandb-config",
            "WANDB_DATA_DIR": args.scratch + "/wandb-data",
            "RAY_USAGE_STATS_CONFIG_PATH": args.scratch + "/ray-usage.json",
            "HF_HOME": args.scratch + "/hf",
            "TRITON_CACHE_DIR": args.scratch + "/triton",
            "TORCHINDUCTOR_CACHE_DIR": args.scratch + "/inductor",
            "CUDA_CACHE_PATH": args.scratch + "/cuda",
            "XDG_CACHE_HOME": args.scratch + "/cache",
            "SGLANG_CACHE_DIR": args.scratch + "/sglang",
            "FLASHINFER_WORKSPACE_BASE": args.scratch + "/flashinfer",
            "TORCH_EXTENSIONS_DIR": args.scratch + "/torch-extensions",
            "MILES_LOG_DIR": args.scratch + "/logs",
            "OMP_NUM_THREADS": "1",
            "PYTHONUNBUFFERED": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    os.chdir(args.repo)
    if args.mode == "preflight":
        _preflight(args)
        return
    start = [
        "ray",
        "start",
        "--num-gpus",
        "8",
        "--disable-usage-stats",
        "--object-store-memory",
        str(args.ray_object_store_gb * 1024**3),
    ]
    if args.role == "worker":
        deadline = time.monotonic() + 180
        while True:
            try:
                with socket.create_connection((args.head_ip, args.ray_port), timeout=2):
                    break
            except OSError as error:
                if time.monotonic() >= deadline:
                    raise RuntimeError("Ray head did not become reachable") from error
                time.sleep(1)
        _run(start + ["--address", f"{args.head_ip}:{args.ray_port}", "--block"])
        return
    _run(
        start
        + [
            "--head",
            "--node-ip-address",
            args.head_ip,
            "--port",
            str(args.ray_port),
            "--dashboard-host",
            "0.0.0.0",
            "--dashboard-port",
            str(args.dashboard_port),
            "--temp-dir",
            f"/tmp/miles-ray-{os.environ['PBS_JOBID'].split('.')[0]}",
        ]
    )
    try:
        _preflight(args)
        _execute_recipe(args)
    finally:
        # This runs inside a dedicated PID namespace, never against another job.
        subprocess.run(["ray", "stop", "--force"], check=False)


def _host(args):
    nodefile = os.environ.get("PBS_NODEFILE")
    if not nodefile or not os.environ.get("PBS_JOBID"):
        raise SystemExit("Run inside a PBS allocation; never launch GPU workloads on a login node")
    slots = Path(nodefile).read_text().split()
    nodes = list(dict.fromkeys(slots))
    if min(args.actor_nodes, args.rollout_nodes, args.num_rollout) < 1:
        raise SystemExit("actor-nodes, rollout-nodes and num-rollout must be positive")
    expected = args.actor_nodes + args.rollout_nodes if args.mode == "train" else 1
    if len(nodes) != expected:
        raise SystemExit(f"{args.mode} requires {expected} node(s), found {len(nodes)}")
    for path in (args.image, args.repo, args.model_dir, args.data_dir, args.judge_config):
        if not Path(path).exists():
            raise SystemExit(f"Missing staged input: {path}")
    if args.container_root and not (Path(args.container_root) / "Megatron-LM").is_dir():
        raise SystemExit("--container-root must contain the pinned image's Megatron-LM checkout")
    Path(args.scratch).mkdir(parents=True, exist_ok=True)
    head_ip = socket.gethostbyname(nodes[0])
    # Bind explicit paths, including /home, while preserving /root container packages.
    command = [
        "singularity",
        "exec",
        "--nv",
        "--pid",
        "--cleanenv",
        "--no-home",
        "--bind",
        "/home:/home",
        "--bind",
        "/groups:/groups:ro",
        "--bind",
        args.scratch + ":" + args.scratch,
        "--env",
        "PBS_JOBID=" + os.environ["PBS_JOBID"],
    ]
    if args.checkpoint_dir:
        checkpoint = Path(args.checkpoint_dir)
        if not checkpoint.is_absolute():
            raise SystemExit("--checkpoint-dir must be absolute for a multi-node bind")
        checkpoint.mkdir(parents=True, exist_ok=True)
        # Only the explicitly selected checkpoint directory is writable under /groups.
        command += ["--bind", f"{checkpoint}:{checkpoint}:rw"]
    if args.container_root:
        command += ["--bind", args.container_root + ":/root:ro"]
    command += [args.image, "python", str(Path(args.repo) / "tools/abci/launch.py")]
    common = [x for x in sys.argv[1:]]
    workers = []
    try:
        for node in nodes[1:]:
            # PBS starts workers in this job's allocation. Ray workers retry until
            # the head comes up; use an explicit wait wrapper instead of SSH agents.
            workers.append(
                subprocess.Popen(
                    ["pbsdsh", "-n", str(slots.index(node)), "--"]
                    + command
                    + common
                    + ["--role", "worker", "--head-ip", head_ip]
                )
            )
        _run_head(command + common + ["--role", "head", "--head-ip", head_ip], workers)
    finally:
        for worker in workers:
            worker.terminate()
            worker.wait()


if __name__ == "__main__":
    args = _parser().parse_args()
    _host(args) if args.role == "host" else _container_role(args)
