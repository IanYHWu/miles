"""Run with torchrun --standalone --nproc-per-node=2 in the pinned Miles image."""

import shutil
import tempfile
from pathlib import Path

import torch
import torch.distributed as dist
from megatron.core import dist_checkpointing
from megatron.core.dist_checkpointing.mapping import ShardedObject, ShardedTensor

from miles.backends.megatron_utils.streaming_checkpoint import StreamingTorchDistSaveStrategy


def main():
    dist.init_process_group("gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    paths = [tempfile.mkdtemp(prefix="miles-checkpoint-test-") if rank == 0 else None]
    dist.broadcast_object_list(paths, src=0)
    path = Path(paths[0])
    expected = torch.arange(6, dtype=torch.float32).reshape(2, 3) + rank * 6

    def state(empty=False):
        return {
            "weight": ShardedTensor.from_rank_offsets(
                "model.weight", torch.zeros_like(expected) if empty else expected, (0, rank, world)
            ),
            "replicated": ShardedTensor.from_rank_offsets(
                "model.replicated", torch.zeros(2) if empty else torch.tensor([3.0, 7.0]), replica_id=rank
            ),
            "rank_state": ShardedObject("rank_state", None if empty else {"rank": rank}, (world,), (rank,)),
            "iteration": 3,
        }

    try:
        dist_checkpointing.save(state(), path, sharded_strategy=StreamingTorchDistSaveStrategy())
        # Use Megatron's default loader, not a matching custom loader.
        restored = dist_checkpointing.load(state(empty=True), path)
        torch.testing.assert_close(restored["weight"], expected, rtol=0, atol=0)
        torch.testing.assert_close(restored["replicated"], torch.tensor([3.0, 7.0]), rtol=0, atol=0)
        assert restored["rank_state"] == {"rank": rank}
        assert restored["iteration"] == 3
        dist.barrier()
        if rank == 0:
            print("Streaming checkpoint: two-rank shards, replicas, objects and default-loader roundtrip passed")
    finally:
        dist.destroy_process_group()
        if rank == 0:
            shutil.rmtree(path)


if __name__ == "__main__":
    main()
