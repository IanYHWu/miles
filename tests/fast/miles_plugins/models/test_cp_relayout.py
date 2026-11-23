"""The CP relayout's sparse all-to-all plan is the same permutation as the all-gather index maps.

Every rank's plan is built on CPU and the all-to-all is simulated, so no process group is needed."""

import pytest
import torch

from miles_plugins.models.cp_utils import _a2a_plan, _relayout_indices

LAYOUTS = [
    ([0, 16], 2),  # one zigzag sequence
    ([0, 8, 24, 30], 2),  # zigzag sequences plus a contiguous-split padding tail (6 % 4 != 0)
    ([0, 24, 40, 64], 4),
    ([0, 4, 12], 4),  # a sequence shorter than 2 * cp_size
]


def _simulate(shards, plans, to_packed):
    """Run every rank's half of the all-to-all and return what each rank ends up holding."""
    cp_size = len(shards)
    sent = []
    for rank, x in enumerate(shards):
        send_index, send_counts, recv_order, recv_counts = plans[rank]
        rows, counts = (send_index, send_counts) if to_packed else (recv_order, recv_counts)
        sent.append(list(x.index_select(0, rows).split(counts)))
    out = []
    for rank in range(cp_size):
        send_index, _, recv_order, _ = plans[rank]
        received = torch.cat([sent[src][rank] for src in range(cp_size)])
        placed = torch.empty_like(received)
        placed.index_copy_(0, recv_order if to_packed else send_index, received)
        out.append(placed)
    return out


@pytest.mark.parametrize("cu,cp_size", LAYOUTS)
def test_the_all_to_all_plan_matches_the_all_gather_relayout(cu, cp_size):
    cu_seqlens = torch.tensor(cu)
    total = cu[-1]
    stream = torch.arange(total * 3, dtype=torch.float32).view(total, 3)  # the packed stream
    packed = list(stream.chunk(cp_size))
    zigzag = [stream.index_select(0, _relayout_indices(cu_seqlens, r, cp_size, total)[1]) for r in range(cp_size)]
    plans = [_a2a_plan(cu_seqlens, r, cp_size, total) for r in range(cp_size)]

    for rank, (got, want) in enumerate(zip(_simulate(zigzag, plans, to_packed=True), packed, strict=True)):
        assert torch.equal(got, want), f"zigzag -> packed differs on rank {rank}"
    for rank, (got, want) in enumerate(zip(_simulate(packed, plans, to_packed=False), zigzag, strict=True)):
        assert torch.equal(got, want), f"packed -> zigzag differs on rank {rank}"


def test_every_row_is_sent_once():
    cu_seqlens, cp_size, total = torch.tensor([0, 8, 24, 30]), 2, 30
    plans = [_a2a_plan(cu_seqlens, r, cp_size, total) for r in range(cp_size)]
    for send_index, send_counts, recv_order, recv_counts in plans:
        assert sorted(send_index.tolist()) == list(range(total // cp_size))
        assert sum(send_counts) == sum(recv_counts) == total // cp_size
        assert sorted(recv_order.tolist()) == list(range(total // cp_size))
