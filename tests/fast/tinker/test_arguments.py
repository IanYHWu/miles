"""Full fine-tuning refuses at startup what its gradient window and per-model optimizer reset cannot honor, and a
gateway that trains nothing takes miles' rollout-only layout."""

from argparse import Namespace

import pytest

from miles.tinker.arguments import configure_tinker_args


def _full_finetune_args(**overrides) -> Namespace:
    defaults = dict(
        tinker_train_mode="full",
        train_backend="megatron",
        exclude_modules=None,
        target_modules=None,
        multi_lora_n_adapters=0,
        lora_rank=0,
        lora_adapter_path=None,
        optimizer="adam",
        debug_disable_optimizer=False,
        use_dynamic_batch_size=True,
        check_for_nan_in_loss_and_grad=False,
        decoupled_lr=None,
        no_save_optim=None,
        update_weight_transfer_mode="broadcast",
        bf16=True,
        fp16=False,
        calculate_per_token_loss=False,
        pause_generation_mode="retract",
        overlap_grad_reduce=False,
        overlap_param_gather=False,
        debug_train_only=False,
        debug_rollout_only=False,
        debug_skip_weight_update=False,
        colocate=False,
        offload=False,
        offload_train=False,
        use_precision_aware_optimizer=False,
        optimizer_cpu_offload=False,
        offload_optimizer_states=False,
        stream_optimizer_state_to_disk=False,
        chunked_optimizer_state_offload=False,
    )
    return Namespace(**{**defaults, **overrides})


def test_full_finetune_takes_no_adapter_targets():
    args = _full_finetune_args()
    configure_tinker_args(args)
    assert not hasattr(args, "tinker_lora_groups")


@pytest.mark.parametrize(
    "overrides,message",
    [
        (dict(multi_lora_n_adapters=4), "drop --multi-lora-n-adapters"),
        (dict(lora_rank=16), "the LoRA flags"),
        (dict(target_modules="attn"), "--target-modules"),
        (dict(optimizer="sgd"), "Adam optimizer"),
        (dict(debug_disable_optimizer=True), "Adam optimizer"),
        (dict(use_dynamic_batch_size=False), "--use-dynamic-batch-size"),
        (dict(check_for_nan_in_loss_and_grad=True), "--no-check-for-nan-in-loss-and-grad"),
        (dict(bf16=False, fp16=True), "use --bf16"),
        (dict(calculate_per_token_loss=True), "--calculate-per-token-loss"),
        (dict(pause_generation_mode="in_place"), "in_place"),
    ],
    ids=[
        "adapter-slots",
        "lora-rank",
        "target-modules",
        "sgd",
        "no-optimizer",
        "static-batches",
        "nan-check",
        "fp16",
        "per-token-loss",
        "in-place-pause",
    ],
)
def test_full_finetune_refuses_conflicting_settings(overrides, message):
    with pytest.raises(AssertionError, match=message):
        configure_tinker_args(_full_finetune_args(**overrides))


@pytest.mark.parametrize(
    "flag",
    [
        "overlap_grad_reduce",
        "overlap_param_gather",
        "debug_train_only",
        "debug_rollout_only",
        "debug_skip_weight_update",
        "colocate",
        "offload",
        "offload_train",
        "use_precision_aware_optimizer",
        "optimizer_cpu_offload",
        "offload_optimizer_states",
        "stream_optimizer_state_to_disk",
        "chunked_optimizer_state_offload",
    ],
)
def test_full_finetune_names_every_unsupported_option(flag):
    with pytest.raises(ValueError, match=f"--{flag.replace('_', '-')}"):
        configure_tinker_args(_full_finetune_args(**{flag: True}))


@pytest.mark.parametrize(
    "overrides,name",
    [
        (dict(decoupled_lr=1e-5), "--decoupled-lr"),
        (dict(no_save_optim=True), "--no-save-optim"),
        (dict(update_weight_transfer_mode="disk-delta"), "--update-weight-transfer-mode disk-delta"),
    ],
    ids=["decoupled-lr", "no-save-optim", "disk-delta"],
)
def test_full_finetune_names_unsupported_values(overrides, name):
    with pytest.raises(ValueError, match=name):
        configure_tinker_args(_full_finetune_args(**overrides))


def test_a_gateway_that_trains_nothing_takes_the_rollout_only_layout():
    args = _full_finetune_args(tinker_train_mode="none")
    configure_tinker_args(args)
    assert args.debug_rollout_only and not hasattr(args, "tinker_lora_groups")


@pytest.mark.parametrize(
    "overrides",
    [dict(multi_lora_n_adapters=4), dict(lora_rank=16), dict(target_modules="attn")],
    ids=["adapter-slots", "lora-rank", "target-modules"],
)
def test_a_gateway_that_trains_nothing_refuses_adapter_settings(overrides):
    with pytest.raises(AssertionError, match="samples only the base model"):
        configure_tinker_args(_full_finetune_args(tinker_train_mode="none", **overrides))
