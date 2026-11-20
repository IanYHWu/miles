from miles.utils.lora.hf_lora_targets import LORA_TARGET_GROUPS, parse_lora_targets
from miles.utils.lora.utils import is_lora_enabled


def add_tinker_arguments(parser):
    group = parser.add_argument_group("Tinker")

    def add_argument(name, **kwargs):
        return group.add_argument(f"--tinker-{name}", **kwargs)

    add_argument("server-host", default="0.0.0.0")
    add_argument("server-port", type=int, default=10613)
    add_argument(
        "base-model",
        help="Model name advertised by the gateway (default: --hf-checkpoint)",
    )
    add_argument(
        "checkpoint-root",
        help="Directory for tinker:// checkpoints (default: <save>/tinker)",
    )
    add_argument(
        "train-mode",
        choices=["lora", "full"],
        default="lora",
        help="What a training client trains: a LoRA adapter in one of --multi-lora-n-adapters slots, "
        "or the served model's own weights, for one client at a time",
    )
    return parser


def configure_tinker_args(args):
    assert args.train_backend == "megatron", "Tinker requires the Megatron backend"
    assert args.exclude_modules is None, "Tinker selects complete training groups; --exclude-modules is not supported"
    if args.tinker_train_mode == "full":
        _validate_full_finetune_args(args)
        return
    groups = parse_lora_targets(args.target_modules)
    if groups is None:
        groups = list(LORA_TARGET_GROUPS)
    assert set(groups) <= set(LORA_TARGET_GROUPS), "Tinker --target-modules accepts only attn,mlp,unembed groups"
    args.tinker_lora_groups = groups
    args.target_modules = groups


def _validate_full_finetune_args(args) -> None:
    """Full fine-tuning keeps one gradient window across commands and reloads its optimizer for each new model."""
    assert not args.multi_lora_n_adapters and not is_lora_enabled(args) and args.target_modules is None, (
        "--tinker-train-mode full trains the served weights; drop --multi-lora-n-adapters, --target-modules and "
        "the LoRA flags"
    )
    assert (
        args.optimizer or "adam"
    ).lower() == "adam" and not args.debug_disable_optimizer, (
        "--tinker-train-mode full steps an Adam optimizer under each request's AdamParams"
    )
    assert args.use_dynamic_batch_size, (
        "--tinker-train-mode full needs --use-dynamic-batch-size --max-tokens-per-gpu: a work unit's datums must "
        "all fit its micro-batches"
    )
    assert not args.check_for_nan_in_loss_and_grad, (
        "--tinker-train-mode full skips an optimizer step on non-finite gradients, as the adapter slots do, where "
        "Megatron's NaN check would end the trainer; pass --no-check-for-nan-in-loss-and-grad"
    )
    assert args.bf16 and not args.fp16, "--tinker-train-mode full runs work units without a loss scaler; use --bf16"
    assert not args.calculate_per_token_loss, "Tinker losses are sums over datums; drop --calculate-per-token-loss"
    assert (
        args.pause_generation_mode != "in_place"
    ), "a full-weight push invalidates every cached KV entry, which --pause-generation-mode in_place keeps"
    unsupported = {
        # AdamParams carry one learning rate, which a per-group one would override
        "--decoupled-lr": args.decoupled_lr is not None,
        # save_state must hold the optimizer state load_state restores
        "--no-save-optim": args.no_save_optim,
        # its first push only records a baseline, and the gateway pushes nothing at startup
        "--update-weight-transfer-mode disk-delta": args.update_weight_transfer_mode == "disk-delta",
        # these reduce gradients during each backward pass; a window reduces them once, at its optimizer step
        "--overlap-grad-reduce": args.overlap_grad_reduce,
        "--overlap-param-gather": args.overlap_param_gather,
        # a sampler save is a weight push, which these skip
        "--debug-train-only": args.debug_train_only,
        "--debug-rollout-only": args.debug_rollout_only,
        "--debug-skip-weight-update": args.debug_skip_weight_update,
        # these move the gradients the window keeps on the GPUs between commands; --offload sets --offload-train later
        "--colocate": args.colocate,
        "--offload": args.offload,
        "--offload-train": args.offload_train,
        # a new model reloads the initial weights with a fresh optimizer, which miles cannot do under these
        "--use-precision-aware-optimizer": args.use_precision_aware_optimizer,
        "--optimizer-cpu-offload": args.optimizer_cpu_offload,
        "--offload-optimizer-states": args.offload_optimizer_states,
        "--stream-optimizer-state-to-disk": args.stream_optimizer_state_to_disk,
        "--chunked-optimizer-state-offload": args.chunked_optimizer_state_offload,
    }
    if names := [name for name, enabled in unsupported.items() if enabled]:
        raise ValueError(f"--tinker-train-mode full does not support {', '.join(names)}")
