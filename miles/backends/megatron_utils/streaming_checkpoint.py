"""Synchronous model checkpoints without a full host staging copy."""

import torch.distributed.checkpoint as dcp
from megatron.core.dist_checkpointing.strategies.torch import (
    MCoreSavePlanner,
    TorchDistSaveShardedStrategy,
    _replace_state_dict_keys_with_sharded_keys,
    mcore_to_pyt_state_dict,
)


class StreamingTorchDistSaveStrategy(TorchDistSaveShardedStrategy):
    def save(self, sharded_state_dict, checkpoint_dir):
        # Preserve Megatron's keys, shard layout and planner. Its default sync
        # writer delegates to an async writer that preloads the entire model.
        translated, _, _ = _replace_state_dict_keys_with_sharded_keys(sharded_state_dict, self.keep_only_main_replica)
        dcp.save(
            mcore_to_pyt_state_dict(translated, False),
            storage_writer=dcp.FileSystemWriter(checkpoint_dir, thread_count=1, per_thread_copy_ahead=0),
            planner=MCoreSavePlanner(
                dedup_replicated_tensors=not self.keep_only_main_replica,
                flatten_state_dict=False,
                flatten_sharded_tensors=False,
            ),
        )
