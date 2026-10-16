"""Synthetic reward for the bounded GPU smoke only; never a training baseline."""


async def reward(args, sample, **kwargs):
    # Alternating sibling rewards guarantee nonzero GRPO advantages even when
    # the 128-token smoke output has not yet closed its thinking block.
    return {"score": float(sample.index % 2)}
