"""Synthetic reward for the bounded GPU smoke of scripts/run_qwen3_5_judge.py; never a training reward."""


async def reward(args, sample, **kwargs):
    # Alternating sibling rewards guarantee nonzero GRPO advantages even when
    # the 128-token smoke output has not yet closed its thinking block.
    return {"score": float(sample.index % 2)}
