import argparse

import pytest

from miles.utils.arguments import get_miles_extra_args_provider, miles_validate_args


def _args(extra=()):
    parser = argparse.ArgumentParser()
    get_miles_extra_args_provider()(parser)
    return parser.parse_args(["--rollout-batch-size", "64", "--num-rollout", "1", *extra])


def test_cispo_is_opt_in_and_preserves_ppo_default():
    args = _args()
    miles_validate_args(args)
    assert args.policy_loss_type == "ppo"
    args = _args(["--policy-loss-type", "cispo", "--eps-clip", "0.2"])
    miles_validate_args(args)
    assert args.eps_clip_high == 0.2


@pytest.mark.parametrize(
    "overrides,message",
    [
        ({"advantage_estimator": "gspo"}, "token-level"),
        ({"eps_clip_c": 3.0}, "Dual clipping"),
        ({"loss_type": "sft_loss"}, "policy_loss"),
        ({"eps_clip": -0.1}, "nonnegative"),
    ],
)
def test_cispo_rejects_incompatible_objectives(overrides, message):
    args = _args(["--policy-loss-type", "cispo"])
    for key, value in overrides.items():
        setattr(args, key, value)
    with pytest.raises(AssertionError, match=message):
        miles_validate_args(args)


@pytest.mark.parametrize("no_save_optim,async_save", [(False, False), (True, True)])
def test_streaming_save_rejects_unsupported_checkpoint_modes(no_save_optim, async_save):
    args = _args(["--streaming-model-checkpoint-save"])
    args.train_backend = "megatron"
    args.ckpt_format = "torch_dist"
    args.no_save_optim = no_save_optim
    args.async_save = async_save
    with pytest.raises(AssertionError, match="synchronous model-only"):
        miles_validate_args(args)
