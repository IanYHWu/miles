import pytest
import torch
from tests.fast.backends.training_utils.loss.loss_test_utils import make_args, make_parallel_state

from miles.backends.training_utils.data.context_parallel import get_sum_of_sample_mean
from miles.backends.training_utils.loss.hub import losses as losses_module
from miles.backends.training_utils.loss.hub.math_utils import compute_cispo_loss


@pytest.mark.parametrize("clip_low,clip_high", [(10.0, 4.0), (0.2, 0.28)])
def test_cispo_matches_reference_gradient_with_padding_and_unequal_lengths(clip_low, clip_high):
    # Two unequal response lengths; include positive/negative advantages,
    # clipped/unclipped ratios, and extreme but finite log ratios.
    current = torch.tensor([-1.0, -2.0, -3.0, -4.0, -1.0, -20.0, -0.1], requires_grad=True)
    delta = torch.tensor([0.0, 2.0, -2.0, 0.5, 30.0, -30.0, 1.0])
    old = (current.detach() - delta).requires_grad_(True)
    advantages = torch.tensor([1.0, -2.0, 3.0, -4.0, 2.0, 1.0, 0.0])
    mask = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0])
    losses, clipfrac = compute_cispo_loss(current, old - current, advantages, clip_low, clip_high)
    loss = (losses * mask).sum() / mask.sum()
    loss.backward()
    # Independent analytical derivative of the reference detached-weight objective.
    weights = delta.clamp(-20, 20).exp().clamp(1 - clip_low, 1 + clip_high)
    expected = -weights * advantages * mask / mask.sum()
    torch.testing.assert_close(current.grad, expected)
    assert old.grad is None
    torch.testing.assert_close(loss.detach(), (expected * current.detach()).sum())
    assert clipfrac[1] == 1


def test_cispo_gradient_is_not_ppo_ratio_gradient():
    logp = torch.tensor([-2.0], requires_grad=True)
    loss, _ = compute_cispo_loss(logp, torch.tensor([-2.0]) - logp, torch.ones(1), 10.0, 4.0)
    loss.sum().backward()
    torch.testing.assert_close(logp.grad, torch.tensor([-1.0]))


def test_policy_dispatch_preserves_masked_token_sum(monkeypatch):
    make_parallel_state()
    args = make_args(
        policy_loss_type="cispo",
        eps_clip=10.0,
        eps_clip_high=4.0,
        use_rollout_logprobs=True,
        skip_actor_forward_only=True,
        entropy_coef=0.0,
        observe_training_entropy=False,
        calculate_per_token_loss=True,
    )
    current = torch.tensor([-1.0, -2.0, -3.0, float("-inf")], requires_grad=True)
    delta = torch.tensor([0.0, 2.0, -2.0, 0.0])
    old = current.detach() - delta
    advantages = torch.tensor([1.0, -2.0, 3.0, float("nan")])
    masks = [torch.ones(1), torch.tensor([1.0, 1.0, 0.0])]
    batch = dict(
        advantages=[advantages[:1], advantages[1:]],
        rollout_log_probs=[old[:1], old[1:]],
        total_lengths=[3, 5],
        response_lengths=[1, 3],
        loss_masks=masks,
        unconcat_tokens=[],
    )
    monkeypatch.setattr(
        losses_module,
        "get_log_probs_and_entropy",
        lambda *a, **kw: {
            "log_probs": [current[:1], current[1:]],
        },
    )
    reducer = get_sum_of_sample_mean([3, 5], [1, 3], masks, True, "thd", None)
    # Miles token mode returns a sum; the training schedule divides by the
    # distributed active-token count. Include that normalization explicitly.
    loss, metrics = losses_module.policy_loss_function(args, batch, torch.zeros(1), reducer)
    (loss / 3).backward()
    expected = -delta[:3].exp().clamp(max=5) * advantages[:3] / 3
    torch.testing.assert_close(current.grad[:3], expected)
    assert current.grad[3] == 0
    assert torch.isfinite(metrics["loss"])
