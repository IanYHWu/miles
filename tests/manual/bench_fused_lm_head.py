"""Time and memory of the LM head plus per-token log-probs, per log-prob backend, on one GPU.

The GPU holds one TP shard of the output layer (``--vocab-shard`` rows of a Megatron
``ColumnParallelLinear`` with an fp32 ``main_grad``), so the TP statistic all-reduces are absent and
every number is per rank. ``torch`` and ``fused`` project every row of the microbatch and keep the
logits; ``fused_lm_head`` projects the scored rows a tile at a time. A train step is the projection,
the scores, a backward of per-token weights on log-probs and entropy, and the weight-gradient
accumulation; a no-grad step only scores. Times are medians after warmup, synchronized; memory is the
peak above the inputs, the layer and its ``main_grad``.

    python tests/manual/bench_fused_lm_head.py [--rows 16384 65536] [--hidden 3072] [--vocab-shard 62080]
"""

import argparse
import gc
import os
import tempfile

import torch
import torch.distributed as dist
from megatron.core import parallel_state as mpu
from megatron.core.model_parallel_config import ModelParallelConfig
from megatron.core.tensor_parallel import ColumnParallelLinear
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.training.global_vars import set_args
from tests.fast.backends.training_utils.loss.loss_test_utils import make_args

from miles.backends.megatron_utils.lm_head import LMHeadCapture
from miles.backends.megatron_utils.parallel import create_megatron_parallel_state
from miles.backends.training_utils.loss.hub.logit_processors import get_log_probs_and_entropy
from miles.backends.training_utils.parallel import GroupInfo, get_parallel_state, set_parallel_state

QWEN35_VOCAB = 248_320
TEMPERATURE = 1.0


def _median_ms(fn, warmup: int = 2, iters: int = 5) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return sorted(times)[len(times) // 2]


def _peak(fn) -> tuple[float, float]:
    """Peak allocated and reserved GiB above what was allocated before ``fn``."""
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    return (torch.cuda.max_memory_allocated() - base) / 2**30, (torch.cuda.max_memory_reserved() - base) / 2**30


def _init_world() -> None:
    with tempfile.NamedTemporaryFile(delete=False) as rendezvous:
        path = rendezvous.name
    os.remove(path)
    dist.init_process_group("nccl", init_method=f"file://{path}", rank=0, world_size=1)
    mpu.initialize_model_parallel(tensor_model_parallel_size=1)
    model_parallel_cuda_manual_seed(1)
    set_args(argparse.Namespace(cp_comm_type=None))
    set_parallel_state(create_megatron_parallel_state(indep_dp=GroupInfo(rank=0, size=1, group=None)))


def _layer(hidden: int, vocab_shard: int) -> ColumnParallelLinear:
    config = ModelParallelConfig(params_dtype=torch.bfloat16, bf16=True, gradient_accumulation_fusion=True)
    layer = ColumnParallelLinear(
        hidden,
        vocab_shard,
        config=config,
        init_method=lambda weight: torch.nn.init.normal_(weight, std=0.02),
        bias=False,
        gather_output=False,
        tp_group=get_parallel_state().tp.group,
    )
    layer.weight.main_grad = torch.zeros_like(layer.weight, dtype=torch.float32)
    layer.weight.grad_added_to_main_grad = False
    return layer


class _Case:
    """One backend's train and no-grad steps over one microbatch of ``rows`` scored tokens."""

    def __init__(self, layer, hidden_values, tokens, *, backend, logits_dtype="model", tile_rows=-1):
        self.layer, self.hidden_values, self.tokens = layer, hidden_values, tokens
        self.backend, self.logits_dtype = backend, logits_dtype
        self.args = make_args(
            true_on_policy_mode=False,
            log_probs_backend=backend,
            log_probs_chunk_size=tile_rows if backend == "fused_lm_head" else 4096,
            rollout_temperature=TEMPERATURE,
            vocab_size=min(QWEN35_VOCAB, layer.weight.size(0)),
        )
        rows = tokens[0].numel() - 1
        gen = torch.Generator(device="cuda").manual_seed(0)
        self.g_log_probs = torch.randn(rows, device="cuda", generator=gen)
        self.g_entropy = torch.randn(rows, device="cuda", generator=gen) * 0.01

    def _output(self, hidden):
        if self.backend == "fused_lm_head":
            capture = LMHeadCapture(logits_dtype=self.logits_dtype, tile_rows=self.args.log_probs_chunk_size)
            capture.record(self.layer, None, hidden)
            return capture.inputs(hidden)
        logits, _ = self.layer(hidden)
        return logits.transpose(0, 1).contiguous()

    def _scores(self, hidden):
        rows = self.tokens[0].numel() - 1
        return get_log_probs_and_entropy(
            self._output(hidden),
            args=self.args,
            unconcat_tokens=self.tokens,
            total_lengths=[rows + 1],
            response_lengths=[rows],
            with_entropy=True,
        )

    def train_step(self) -> None:
        self.layer.weight.grad_added_to_main_grad = False
        self.layer.weight.grad = None
        hidden = self.hidden_values.detach().requires_grad_(True)
        res = self._scores(hidden)
        loss = (torch.cat(res["log_probs"]) * self.g_log_probs).sum() + (
            torch.cat(res["entropy"]) * self.g_entropy
        ).sum()
        loss.backward()

    def no_grad_step(self) -> None:
        with torch.no_grad():
            self._scores(self.hidden_values)


def _measure(name: str, case: _Case) -> None:
    results = []
    for step in (case.train_step, case.no_grad_step):
        try:
            ms = _median_ms(step)
            allocated, reserved = _peak(step)
            results.append(f"{ms:8.1f} ms {allocated:6.2f} / {reserved:6.2f} GiB")
        except torch.OutOfMemoryError:
            results.append(f"{'OOM':>36}")
        gc.collect()
        torch.cuda.empty_cache()
    print(f"  {name:<28} train {results[0]}   no-grad {results[1]}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, nargs="+", default=[16384, 65536])
    parser.add_argument("--hidden", type=int, default=3072)
    parser.add_argument("--vocab-shard", type=int, default=QWEN35_VOCAB // 4)
    parser.add_argument("--tiles", type=int, nargs="+", default=[1024, 4096, 16384])
    parser.add_argument("--skip-torch", action="store_true", help="the torch backend's fp32 copies may not fit")
    args = parser.parse_args()

    _init_world()
    matmul = torch.backends.cuda.matmul
    print(
        f"{torch.cuda.get_device_name()}, torch {torch.__version__}; hidden {args.hidden}, vocab shard "
        f"{args.vocab_shard}; allow_bf16_reduced_precision_reduction={matmul.allow_bf16_reduced_precision_reduction}, "
        f"allow_tf32={matmul.allow_tf32}"
    )
    print("  backend                      train: median, peak allocated / reserved above inputs; no-grad: same")
    layer = _layer(args.hidden, args.vocab_shard)
    for rows in args.rows:
        gen = torch.Generator(device="cuda").manual_seed(rows)
        hidden_values = torch.randn(rows + 1, 1, args.hidden, device="cuda", generator=gen).to(torch.bfloat16)
        vocab = min(QWEN35_VOCAB, args.vocab_shard)
        tokens = [torch.randint(0, vocab, (rows + 1,), device="cuda", generator=gen)]
        materialized = rows * args.vocab_shard * 2 / 2**30
        print(f"{rows} scored rows (bf16 logits of the microbatch: {materialized:.2f} GiB)")
        cases = [] if args.skip_torch else [("torch", _Case(layer, hidden_values, tokens, backend="torch"))]
        cases.append(("fused", _Case(layer, hidden_values, tokens, backend="fused")))
        for logits_dtype in ("model", "fp32"):
            for tile_rows in args.tiles:
                case = _Case(
                    layer,
                    hidden_values,
                    tokens,
                    backend="fused_lm_head",
                    logits_dtype=logits_dtype,
                    tile_rows=tile_rows,
                )
                cases.append((f"fused_lm_head {logits_dtype} {tile_rows}", case))
        for name, case in cases:
            _measure(name, case)
    mpu.destroy_model_parallel()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
