"""Collective entry point of the vLLM weight push: every trainer rank calls it, rank 0 sends.

Failures are returned as ``{"ok": False, "error": ...}`` rather than raised: a raise inside a
trainer RPC kills the trainer cell, while a driver can reconnect to the surviving servers and push
again (see docs/advanced/vllm-weight-push.md).
"""

import logging
from collections.abc import Callable, Iterable, Sequence

import torch
import torch.distributed as dist

from miles.backends.training_utils.weight_update.vllm_nccl.pusher import NamedTensors, VllmEngine, VllmPusher
from miles.utils.distributed_utils import get_gloo_group

logger = logging.getLogger(__name__)

_MAX_ERROR_CHARS = 2000


class VllmWeightUpdater:
    """Owns global rank 0's ``VllmPusher``; ``iter_hf_weights`` yields the HF-named weight batches
    collectively on every rank (the other ranks only join the gathers)."""

    def __init__(self, iter_hf_weights: Callable[[], Iterable[NamedTensors]]):
        self._iter_hf_weights = iter_hf_weights
        self._pusher: VllmPusher | None = None

    def connect(
        self,
        engines: Sequence[VllmEngine | dict],
        nccl_so: str,
        *,
        http_timeout: float,
        connect_timeout: float,
        push_timeout: float,
        extra_ranks: int = 0,
        test_batch_delay: float = 0.0,
    ) -> dict | None:
        """Open one NCCL communicator per server, replacing any earlier connection. Rank 0 only:
        the other ranks return ``None`` immediately."""
        if dist.get_rank() != 0:
            return None
        if self._pusher is not None:
            self._pusher.close()
        self._pusher = VllmPusher(
            engines,
            nccl_so,
            http_timeout=http_timeout,
            connect_timeout=connect_timeout,
            extra_ranks=extra_ranks,
            push_timeout=push_timeout,
            test_batch_delay=test_batch_delay,
        )
        try:
            return dict(self._pusher.connect(), ok=True)
        except Exception as error:  # noqa: BLE001 - returned, not raised: a raise kills the trainer cell
            logger.exception("VLLM_PUSH connect failed")
            self._pusher.close()
            self._pusher = None
            return {"ok": False, "error": repr(error)[:_MAX_ERROR_CHARS], "wedged": list(getattr(error, "wedged", []))}

    def push(self, version: str) -> dict | None:
        """Send the current weights to every connected server and publish ``version`` (collective)."""
        torch.cuda.synchronize()
        batches = self._iter_hf_weights()
        if dist.get_rank() != 0:
            for _ in batches:
                pass
            dist.barrier(group=get_gloo_group())
            return None
        stream = iter(batches)
        try:
            if self._pusher is None:
                raise RuntimeError("not connected to any server (vllm_connect failed or was not called)")
            # the pusher's own ok wins: a push with no servers to send to sends nothing and is not ok
            return {"ok": True, **self._pusher.push(stream, version)}
        except Exception as error:  # noqa: BLE001 - returned, not raised: a raise kills the trainer cell
            logger.exception("VLLM_PUSH push failed")
            for _ in stream:  # the other ranks are still in the gather collectives: finish them
                pass
            return {"ok": False, "error": repr(error)[:_MAX_ERROR_CHARS]}
        finally:
            dist.barrier(group=get_gloo_group())
