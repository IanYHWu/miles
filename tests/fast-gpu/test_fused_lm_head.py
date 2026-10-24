"""``--log-probs-backend fused_lm_head`` against materialized logits from the same Megatron output layer.

Each check projects through a real ``ColumnParallelLinear`` carrying a ``main_grad`` buffer, as the
optimizer sees it, over TP1, TP2 and TP2 with sequence parallelism, and compares log-probs, entropy,
the hidden-state gradient and the accumulated weight gradient. ``_exact_problem`` draws hidden states
and weights on a coarse grid, so every bf16 logit is exact whatever the GEMM's summation order: the
model-precision head must then reproduce the materialized logits' scores bit for bit. ``_sensitive_problem``
puts the logits near 320, where bf16 is 2 apart, to catch an fp32 mode that only upcasts bf16 logits.
The MTP loss's tiled cross-entropy is checked against Megatron's on the same layer's materialized logits.
"""

import sys

from tests.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=300, suite="stage-b-2-gpu-h200", labels=["megatron"], hardware=["hopper", "blackwell"], num_gpus=2
)

import dataclasses
import functools
import os
from argparse import Namespace
from collections.abc import Callable

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from megatron.core import parallel_state as mpu
from megatron.core.model_parallel_config import ModelParallelConfig
from megatron.core.tensor_parallel import ColumnParallelLinear, vocab_parallel_cross_entropy
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.training.global_vars import set_args
from tests.fast.backends.training_utils.loss.loss_test_utils import make_args
from tests.fast.dist_utils import find_free_port

from miles.backends.megatron_utils.lm_head import LMHeadCapture
from miles.backends.megatron_utils.parallel import create_megatron_parallel_state
from miles.backends.training_utils.loss.hub.logit_processors import get_log_probs_and_entropy
from miles.backends.training_utils.loss.objective import loss_function
from miles.backends.training_utils.parallel import GroupInfo, get_parallel_state, set_parallel_state

_WORLD_SIZE = 2
_HIDDEN = 64
_VOCAB = 1000  # the true vocabulary; the layer holds Megatron's padded one
_PADDED_VOCAB = 1024
_DELIMITERS = list(range(_VOCAB - 6, _VOCAB))  # the top real ids, like the private tokenizer's added delimiters
_TEMPERATURE = 0.7
_LAYOUTS = [(1, False), (2, False), (2, True)]
_LAYOUT_IDS = ["tp1", "tp2", "tp2_sp"]


# ---------------------------------------------------------------------------
# Distributed setup
# ---------------------------------------------------------------------------


def _worker(rank: int, port: int, tp: int, sequence_parallel: bool, checks: tuple[Callable, ...]) -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=_WORLD_SIZE)
    try:
        mpu.initialize_model_parallel(tensor_model_parallel_size=tp)
        model_parallel_cuda_manual_seed(1234)
        set_args(Namespace(cp_comm_type=None))  # all create_megatron_parallel_state reads from Megatron's args
        _set_parallel_state()
        for check in checks:
            try:
                check(sequence_parallel=sequence_parallel)
            except Exception as error:
                error.add_note(f"in {check.__name__}")
                raise
    finally:
        mpu.destroy_model_parallel()
        dist.destroy_process_group()


def _spawn(tp: int, sequence_parallel: bool, checks: tuple[Callable, ...]) -> None:
    if torch.cuda.device_count() < _WORLD_SIZE:
        raise RuntimeError(f"requires {_WORLD_SIZE} GPUs, found {torch.cuda.device_count()}")
    mp.spawn(_worker, args=(find_free_port(), tp, sequence_parallel, checks), nprocs=_WORLD_SIZE, join=True)


def _set_parallel_state(cp_rank: int = 0, cp_size: int = 1) -> None:
    """Megatron's state, with every rank alone in its data-parallel group: each runs the same check."""
    trivial = GroupInfo(rank=0, size=1, group=None)
    state = create_megatron_parallel_state(indep_dp=trivial)
    set_parallel_state(
        dataclasses.replace(
            state, intra_dp=trivial, intra_dp_cp=trivial, cp=GroupInfo(rank=cp_rank, size=cp_size, group=None)
        )
    )


# ---------------------------------------------------------------------------
# Problems: an output layer, hidden states and the samples they score
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class _Problem:
    layer: ColumnParallelLinear
    full_weight: torch.Tensor  # [padded V, H] bf16, every rank's shard
    hidden: torch.Tensor  # [S, B, H] bf16, the whole sequence
    tokens: list[torch.Tensor]
    total_lens: list[int]
    response_lens: list[int]
    qkv_format: str
    max_seq_lens: list[int] | None

    @property
    def tp(self) -> tuple[int, int]:
        tp = get_parallel_state().tp
        return tp.rank, tp.size

    def weight_shard(self, full: torch.Tensor) -> torch.Tensor:
        rank, size = self.tp
        rows = full.size(0) // size
        return full[rank * rows : (rank + 1) * rows]

    def local_hidden(self) -> torch.Tensor:
        """This rank's model output: the sequence shard under SP, the whole sequence otherwise."""
        if not self.layer.sequence_parallel:
            return self.hidden.detach().clone().requires_grad_(True)
        rank, size = self.tp
        rows = self.hidden.size(0) // size
        return self.hidden[rank * rows : (rank + 1) * rows].detach().clone().requires_grad_(True)

    def reset_grads(self) -> None:
        weight = self.layer.weight
        weight.main_grad.zero_()
        weight.grad_added_to_main_grad = False
        weight.grad = None


def _layer(sequence_parallel: bool, full_weight: torch.Tensor) -> ColumnParallelLinear:
    """A bias-free output layer holding this rank's shard of ``full_weight`` ``[padded V, H]``."""
    padded_vocab, hidden_size = full_weight.shape
    tp = get_parallel_state().tp
    config = ModelParallelConfig(
        tensor_model_parallel_size=tp.size,
        sequence_parallel=sequence_parallel and tp.size > 1,
        params_dtype=torch.bfloat16,
        bf16=True,
        gradient_accumulation_fusion=True,
    )
    layer = ColumnParallelLinear(
        hidden_size,
        padded_vocab,
        config=config,
        init_method=torch.nn.init.zeros_,
        bias=False,
        gather_output=False,
        tp_group=tp.group,
    )
    rows = padded_vocab // tp.size
    with torch.no_grad():
        layer.weight.copy_(full_weight[tp.rank * rows : (tp.rank + 1) * rows])
    # what Megatron's DDP attaches: the fp32 buffer the fused wgrad accumulates into
    layer.weight.main_grad = torch.zeros_like(layer.weight, dtype=torch.float32)
    layer.weight.grad_added_to_main_grad = False
    return layer


def _samples(seed: int, prompt_lens: list[int], response_lens: list[int], qkv_format: str):
    gen = torch.Generator().manual_seed(seed)
    total_lens = [p + r for p, r in zip(prompt_lens, response_lens, strict=True)]
    tokens = []
    for total, response in zip(total_lens, response_lens, strict=True):
        sample = torch.randint(0, _VOCAB, (total,), generator=gen)
        # every response scores some of the highest real ids, which sit next to the vocab padding
        for offset, delimiter in zip(range(0, response, 3), _DELIMITERS, strict=False):
            sample[total - response + offset] = delimiter
        tokens.append(sample.cuda())
    tp_size = get_parallel_state().tp.size
    if qkv_format == "thd":
        seq_len, batch_size, max_seq_lens = _round_up(sum(total_lens), tp_size), 1, None
    else:
        seq_len, batch_size = _round_up(max(total_lens), tp_size), len(total_lens)
        max_seq_lens = [seq_len] * batch_size
    return tokens, total_lens, (seq_len, batch_size), max_seq_lens


def _round_up(value: int, multiple: int) -> int:
    return -(-value // multiple) * multiple


def _exact_problem(sequence_parallel, *, qkv_format="thd", prompt_lens=(5, 9, 3, 4), response_lens=(7, 0, 13, 6)):
    """Grid values: every logit is a multiple of 1/64 below 4 in magnitude, exact in bf16."""
    tokens, total_lens, (seq_len, batch_size), max_seq_lens = _samples(
        11, list(prompt_lens), list(response_lens), qkv_format
    )
    gen = torch.Generator().manual_seed(7)
    full_weight = (torch.randint(-2, 3, (_PADDED_VOCAB, _HIDDEN), generator=gen) * 0.125).to(torch.bfloat16).cuda()
    hidden = (torch.randint(-2, 3, (seq_len, batch_size, _HIDDEN), generator=gen) * 0.125).to(torch.bfloat16).cuda()
    return _Problem(
        _layer(sequence_parallel, full_weight),
        full_weight,
        hidden,
        tokens,
        total_lens,
        list(response_lens),
        qkv_format,
        max_seq_lens,
    )


def _sensitive_problem(sequence_parallel):
    """Logits near 320, where bf16 steps by 2 while the true logits differ by tenths."""
    tokens, total_lens, (seq_len, batch_size), max_seq_lens = _samples(13, [6, 4], [20, 11], "thd")
    gen = torch.Generator().manual_seed(5)
    full_weight = torch.randn(_PADDED_VOCAB, _HIDDEN, generator=gen) * 0.05
    full_weight[:, 0] = 16.0
    hidden = torch.randn(seq_len, batch_size, _HIDDEN, generator=gen) * 0.5
    hidden[..., 0] = 20.0
    full_weight, hidden = full_weight.to(torch.bfloat16).cuda(), hidden.to(torch.bfloat16).cuda()
    return _Problem(
        _layer(sequence_parallel, full_weight), full_weight, hidden, tokens, total_lens, [20, 11], "thd", max_seq_lens
    )


# ---------------------------------------------------------------------------
# The paths under comparison
# ---------------------------------------------------------------------------


def _args(backend: str, **overrides):
    return make_args(
        **{
            "true_on_policy_mode": False,
            "log_probs_backend": backend,
            "rollout_temperature": _TEMPERATURE,
            "vocab_size": _VOCAB,
            "log_probs_chunk_size": 4096,
            "observe_training_entropy": False,
            **overrides,
        }
    )


def _model_output(problem: _Problem, backend: str, hidden: torch.Tensor, logits_dtype: str = "model"):
    """What the loss receives: materialized ``[B, S, V / TP]`` logits, or the forward's LMHeadInputs."""
    if backend == "fused_lm_head":
        capture = LMHeadCapture(logits_dtype=logits_dtype, tile_rows=4096)
        capture.record(problem.layer, None, hidden)
        return capture.inputs(hidden)
    logits, _ = problem.layer(hidden)
    return logits.transpose(0, 1).contiguous()


def _score(problem: _Problem, args, model_output, *, with_entropy=True, entropy_requires_grad=True):
    res = get_log_probs_and_entropy(
        model_output,
        args=args,
        unconcat_tokens=problem.tokens,
        total_lengths=problem.total_lens,
        response_lengths=problem.response_lens,
        with_entropy=with_entropy,
        entropy_requires_grad=entropy_requires_grad,
        max_seq_lens=problem.max_seq_lens,
    )
    return res["log_probs"], res.get("entropy")


@dataclasses.dataclass
class _Run:
    log_probs: torch.Tensor
    entropy: torch.Tensor | None
    hidden_grad: torch.Tensor | None
    main_grad: torch.Tensor
    sizes: list[int]


def _upstream(sizes: list[int], seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token weights of mixed sign with exact zeros, for log-probs and for entropy."""
    gen = torch.Generator().manual_seed(seed)
    total = sum(sizes)
    weights = torch.randn(2, total, generator=gen)
    weights[:, ::5] = 0.0
    return weights[0].cuda(), weights[1].cuda()


def _run(problem: _Problem, backend: str, *, logits_dtype="model", tile_rows=4096, entropy="grad") -> _Run:
    """Score with ``backend`` and backpropagate arbitrary per-token weights of log-probs and entropy."""
    problem.reset_grads()
    hidden = problem.local_hidden()
    args = _args(backend, log_probs_chunk_size=tile_rows, qkv_format=problem.qkv_format)
    with_entropy = entropy != "off"
    log_probs, entropy_list = _score(
        problem,
        args,
        _model_output(problem, backend, hidden, logits_dtype),
        with_entropy=with_entropy,
        entropy_requires_grad=entropy == "grad",
    )
    sizes = [lp.numel() for lp in log_probs]
    log_probs = torch.cat(log_probs)
    ent = torch.cat(entropy_list) if with_entropy else None
    g_log_probs, g_entropy = _upstream(sizes, seed=3)
    loss = (g_log_probs * log_probs).sum()
    if entropy == "grad":
        loss = loss + (g_entropy * ent).sum()
    loss.backward()
    return _Run(
        log_probs.detach(),
        ent.detach() if ent is not None else None,
        hidden.grad,
        problem.layer.weight.main_grad.clone(),
        sizes,
    )


def _reference(problem: _Problem, *, entropy="grad") -> _Run:
    """Fp64 log-softmax over the true vocabulary of fp64 logits from the bf16 operands."""
    weight = problem.full_weight.double().requires_grad_(True)
    hidden_full = problem.hidden.double().requires_grad_(True)
    seq_len, batch_size, _ = problem.hidden.shape
    logits = (hidden_full.transpose(0, 1).reshape(seq_len * batch_size, _HIDDEN) @ weight.t())[:, :_VOCAB]
    log_softmax = F.log_softmax(logits / _TEMPERATURE, dim=-1)
    rows, targets, sizes = [], [], []
    for i, (tokens, total, response) in enumerate(
        zip(problem.tokens, problem.total_lens, problem.response_lens, strict=True)
    ):
        start = (sum(problem.total_lens[:i]) if problem.qkv_format == "thd" else i * seq_len) + total - response - 1
        rows.append(torch.arange(start, start + response))
        targets.append(tokens[total - response :])
        sizes.append(response)
    rows, targets = torch.cat(rows).cuda(), torch.cat(targets)
    selected = log_softmax[rows]
    log_probs = selected.gather(1, targets.unsqueeze(1)).squeeze(1)
    ent = -(selected.exp() * selected).sum(-1)
    g_log_probs, g_entropy = _upstream(sizes, seed=3)
    loss = (g_log_probs.double() * log_probs).sum()
    if entropy == "grad":
        loss = loss + (g_entropy.double() * ent).sum()
    loss.backward()
    hidden_grad = hidden_full.grad
    if problem.layer.sequence_parallel:
        rank, size = problem.tp
        shard = seq_len // size
        hidden_grad = hidden_grad[rank * shard : (rank + 1) * shard]
    return _Run(log_probs.detach(), ent.detach(), hidden_grad, problem.weight_shard(weight.grad), sizes)


def _assert_grads_close(actual: _Run, expected: _Run, *, rtol: float) -> None:
    """Gradients agree to ``rtol`` of their own scale: bf16 rounding, not elementwise exactness."""
    for name in ("hidden_grad", "main_grad"):
        error = _relative_error(getattr(actual, name), getattr(expected, name))
        assert error < rtol, f"{name}: max error {error:.3g} of its scale"


# ---------------------------------------------------------------------------
# Checks, each run on every rank of a spawned world
# ---------------------------------------------------------------------------


def _check_parity(*, sequence_parallel: bool) -> None:
    for qkv_format in ("thd", "bshd"):
        problem = _exact_problem(sequence_parallel, qkv_format=qkv_format)
        materialized = _run(problem, "fused")
        reference = _reference(problem)
        torch.testing.assert_close(materialized.log_probs, reference.log_probs.float(), rtol=1e-5, atol=1e-5)
        for logits_dtype in ("model", "fp32"):
            # one row per tile, an uneven tail of 7, and one tile holding every row
            for tile_rows in (1, 7, 4096):
                run = _run(problem, "fused_lm_head", logits_dtype=logits_dtype, tile_rows=tile_rows)
                where = f"{qkv_format}, {logits_dtype} logits, {tile_rows}-row tiles"
                assert run.sizes == materialized.sizes, where
                # exact bf16 logits: the model-precision tiles feed the materialized logits' kernel the same values,
                # so it agrees bitwise; fp32 tiles hold the same values but run the kernel compiled for fp32 input
                exact = dict(rtol=0, atol=0) if logits_dtype == "model" else dict(rtol=1e-6, atol=1e-6)
                torch.testing.assert_close(run.log_probs, materialized.log_probs, **exact, msg=where)
                torch.testing.assert_close(run.entropy, materialized.entropy, **exact, msg=where)
                _assert_grads_close(run, materialized, rtol=1e-2)
                _assert_grads_close(run, reference, rtol=2e-2)
                rank, size = problem.tp
                padding_rows = max(
                    0, (rank + 1) * (_PADDED_VOCAB // size) - max(_VOCAB, rank * (_PADDED_VOCAB // size))
                )
                if padding_rows:
                    assert (run.main_grad[-padding_rows:] == 0).all(), "padding rows of the weight got a gradient"


def _check_entropy_modes(*, sequence_parallel: bool) -> None:
    problem = _exact_problem(sequence_parallel)
    for entropy in ("off", "detached", "grad"):
        materialized = _run(problem, "fused", entropy=entropy)
        run = _run(problem, "fused_lm_head", tile_rows=5, entropy=entropy)
        torch.testing.assert_close(run.log_probs, materialized.log_probs, rtol=0, atol=0)
        if entropy != "off":
            torch.testing.assert_close(run.entropy, materialized.entropy, rtol=0, atol=0)
        _assert_grads_close(run, materialized, rtol=1e-2)
        _assert_grads_close(run, _reference(problem, entropy=entropy), rtol=2e-2)


def _check_fp32_logits(*, sequence_parallel: bool) -> None:
    problem = _sensitive_problem(sequence_parallel)
    reference = _reference(problem)
    fp32 = _run(problem, "fused_lm_head", logits_dtype="fp32", tile_rows=8)
    model = _run(problem, "fused_lm_head", logits_dtype="model", tile_rows=8)
    # 64 fp32 products around 320 accumulate errors of a few 1e-5
    torch.testing.assert_close(fp32.log_probs, reference.log_probs.float(), rtol=0, atol=1e-3)
    torch.testing.assert_close(fp32.entropy, reference.entropy.float(), rtol=0, atol=1e-3)
    model_error = (model.log_probs - reference.log_probs.float()).abs().max()
    assert model_error > 0.05, f"bf16 logits should be off by tenths here, got {model_error}"
    # the backward's replayed tiles are fp32 too: from bf16 logits the weight gradient would follow `model`.
    # Rows no response targets get only -p * h, so their gradient is the softmax itself.
    untargeted = _untargeted_rows(problem)
    fp32_grad_error, model_grad_error = (
        _relative_error(run.main_grad[untargeted], reference.main_grad[untargeted]) for run in (fp32, model)
    )
    assert fp32_grad_error < 3e-2 and fp32_grad_error < model_grad_error / 4, (fp32_grad_error, model_grad_error)
    assert _relative_error(fp32.main_grad, reference.main_grad) < 3e-2
    # hidden dimension 0 carries the shared 16 * 20 offset; its gradient sums to zero over the vocabulary,
    # so the bf16-rounded logits gradient leaves only cancellation noise there
    assert _relative_error(fp32.hidden_grad[..., 1:], reference.hidden_grad[..., 1:]) < 5e-2


def _untargeted_rows(problem: _Problem) -> torch.Tensor:
    """This rank's weight rows of real tokens that no response scores."""
    rank, size = problem.tp
    rows = _PADDED_VOCAB // size
    token_ids = torch.arange(rank * rows, (rank + 1) * rows, device="cuda")
    targets = torch.cat(
        [
            tokens[total - response :]
            for tokens, total, response in zip(problem.tokens, problem.total_lens, problem.response_lens, strict=True)
        ]
    )
    return (token_ids < _VOCAB) & ~torch.isin(token_ids, targets)


def _relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    expected = expected.double()
    return ((actual.double() - expected).abs().max() / expected.abs().max().clamp_min(1e-12)).item()


def _check_no_grad_scores(*, sequence_parallel: bool) -> None:
    problem = _exact_problem(sequence_parallel)
    for logits_dtype in ("model", "fp32"):
        trained = _run(problem, "fused_lm_head", logits_dtype=logits_dtype, tile_rows=6)
        with torch.no_grad():
            hidden = problem.local_hidden().detach()
            log_probs, entropy = _score(
                problem,
                _args("fused_lm_head", log_probs_chunk_size=6),
                _model_output(problem, "fused_lm_head", hidden, logits_dtype),
            )
        torch.testing.assert_close(torch.cat(log_probs), trained.log_probs, rtol=0, atol=0)
        torch.testing.assert_close(torch.cat(entropy), trained.entropy, rtol=0, atol=0)


def _check_empty_responses(*, sequence_parallel: bool) -> None:
    """A rank that scores nothing must still send the backward through the head and the backbone."""
    problem = _exact_problem(sequence_parallel, prompt_lens=(5, 9), response_lens=(0, 0))
    hidden = problem.local_hidden()
    log_probs, _ = _score(problem, _args("fused_lm_head"), _model_output(problem, "fused_lm_head", hidden))
    assert [lp.numel() for lp in log_probs] == [0, 0]
    batch = _loss_batch(problem, log_probs)
    for name in ("sft", "ppo_entropy_regularized_kl"):
        for backend in ("fused", "fused_lm_head"):
            loss, _, hidden_grad, main_grad = _loss_run(problem, backend, batch, _LOSS_CASES[name])
            where = f"{name} through {backend}"
            assert loss.item() == 0, where
            assert hidden_grad is not None and (hidden_grad == 0).all(), where
            # what Megatron's DDP waits for before it reduces the output layer's gradient bucket
            assert problem.layer.weight.grad_added_to_main_grad, f"{where}: the output weight missed the backward"
            assert (main_grad == 0).all(), where


def _check_zigzag_cp_ranks(*, sequence_parallel: bool) -> None:
    """Zigzag CP2 needs no collective to score its rows, so one process stands in for each CP rank.

    CP rank 1 holds chunks 1 and 2 of the first sample, wholly inside its 90-token prompt.
    """
    prompt_lens, response_lens = [90, 10], [10, 30]
    for cp_rank in (0, 1):
        _set_parallel_state(cp_rank=cp_rank, cp_size=2)
        try:
            totals = [p + r for p, r in zip(prompt_lens, response_lens, strict=True)]
            local_rows = sum(2 * -(-total // 4) for total in totals)
            problem = _exact_problem(sequence_parallel, prompt_lens=prompt_lens, response_lens=response_lens)
            problem.hidden = problem.hidden[:local_rows]
            materialized = _run(problem, "fused")
            run = _run(problem, "fused_lm_head", tile_rows=9)
            assert run.sizes == materialized.sizes
            if cp_rank == 1:
                assert run.sizes[0] == 0
            torch.testing.assert_close(run.log_probs, materialized.log_probs, rtol=0, atol=0)
            _assert_grads_close(run, materialized, rtol=1e-2)
        finally:
            _set_parallel_state()


def _check_rejections(*, sequence_parallel: bool) -> None:
    problem = _exact_problem(sequence_parallel)
    hidden = problem.local_hidden()
    with pytest.raises(ValueError, match="only the fused_lm_head backend scores LMHeadInputs"):
        _score(problem, _args("fused"), _model_output(problem, "fused_lm_head", hidden))
    with pytest.raises(ValueError, match="got Tensor"):
        _score(problem, _args("fused_lm_head"), _model_output(problem, "fused", hidden))
    with pytest.raises(RuntimeError, match="without calling its output_processor"):
        LMHeadCapture(logits_dtype="model", tile_rows=8).inputs(hidden)
    with pytest.raises(ValueError, match="must not upcast its output"):
        LMHeadCapture(logits_dtype="model", tile_rows=8).record(problem.layer, None, hidden.float())
    head = LMHeadCapture(logits_dtype="model", tile_rows=8).mtp_output_layer(problem.layer)
    with pytest.raises(ValueError, match="only returns the per-token cross-entropy"):
        head(hidden)


def _check_frozen_weight(*, sequence_parallel: bool) -> None:
    """A weight that takes no gradient: the same scores and input gradient, and its ``main_grad`` untouched."""
    problem = _exact_problem(sequence_parallel)
    trainable = _run(problem, "fused")
    problem.layer.weight.requires_grad_(False)
    try:
        for logits_dtype in ("model", "fp32"):
            run = _run(problem, "fused_lm_head", logits_dtype=logits_dtype, tile_rows=7)
            exact = dict(rtol=0, atol=0) if logits_dtype == "model" else dict(rtol=1e-6, atol=1e-6)
            torch.testing.assert_close(run.log_probs, trainable.log_probs, **exact, msg=logits_dtype)
            assert _relative_error(run.hidden_grad, trainable.hidden_grad) < 1e-2, logits_dtype
            assert not problem.layer.weight.grad_added_to_main_grad and (run.main_grad == 0).all(), logits_dtype
    finally:
        problem.layer.weight.requires_grad_(True)


def _check_weight_grad_hook(*, sequence_parallel: bool) -> None:
    """The weight's gradient hook, which Megatron's DDP waits on, fires once, after every tile's wgrad.

    Megatron's linear signals it by setting ``grad_added_to_main_grad`` and returning a dummy gradient;
    inside the tiles' checkpoint ``track_weight_grad`` does that for it, for the policy and the MTP head.
    """
    problem = _exact_problem(sequence_parallel)
    seq_len, batch_size, _ = problem.hidden.shape
    gen = torch.Generator().manual_seed(13)
    labels = torch.randint(0, _VOCAB, (batch_size, seq_len), generator=gen).cuda()
    upstream = torch.randn(batch_size, seq_len, generator=gen).cuda()
    seen = []
    handle = problem.layer.weight.register_post_accumulate_grad_hook(
        lambda param: seen.append((param.grad_added_to_main_grad, param.main_grad.clone()))
    )
    try:
        for logits_dtype in ("model", "fp32"):
            runs = {
                f"policy, {tile_rows}-row tiles": functools.partial(
                    _run, problem, "fused_lm_head", logits_dtype=logits_dtype, tile_rows=tile_rows
                )
                for tile_rows in (7, 4096)
            }
            runs["MTP"] = functools.partial(
                _tiled_cross_entropy, problem, labels, upstream, detached=False, logits_dtype=logits_dtype
            )
            for name, run in runs.items():
                seen.clear()
                main_grad = run().main_grad
                where = f"{name}, {logits_dtype} logits"
                assert len(seen) == 1, f"{where}: the weight's gradient hook fired {len(seen)} times"
                added, main_grad_at_hook = seen[0]
                assert added, f"{where}: grad_added_to_main_grad unset when the hook fired"
                torch.testing.assert_close(main_grad_at_hook, main_grad, rtol=0, atol=0, msg=f"{where}: early hook")
                assert main_grad.abs().max() > 0, where
    finally:
        handle.remove()


def _check_mtp_cross_entropy(*, sequence_parallel: bool) -> None:
    """The MTP loss through the tiles against Megatron's cross-entropy of the layer's materialized logits.

    Over the padded vocabulary, as Megatron's is; with the weight trainable and detached, as
    ``--mtp-detach-heads`` passes it; ``ignore_index`` labels cost nothing.
    """
    for qkv_format in ("thd", "bshd"):
        problem = _exact_problem(sequence_parallel, qkv_format=qkv_format)
        seq_len, batch_size, _ = problem.hidden.shape
        gen = torch.Generator().manual_seed(11)
        labels = torch.randint(0, _PADDED_VOCAB, (batch_size, seq_len), generator=gen).cuda()
        labels[:, 3::7] = -100
        upstream = torch.randn(batch_size, seq_len, generator=gen).cuda()
        for detached in (False, True):
            expected = _megatron_cross_entropy(problem, labels, upstream, detached=detached)
            for logits_dtype in ("model", "fp32"):
                actual = _tiled_cross_entropy(problem, labels, upstream, detached=detached, logits_dtype=logits_dtype)
                where = f"{qkv_format}, detached={detached}, {logits_dtype} logits"
                torch.testing.assert_close(actual.log_probs, expected.log_probs, rtol=1e-5, atol=1e-5, msg=where)
                assert (actual.log_probs[labels == -100] == 0).all(), where
                _assert_grads_close(actual, expected, rtol=1e-2)
                if detached:
                    assert (actual.main_grad == 0).all() and not problem.layer.weight.grad_added_to_main_grad, where


def _tiled_cross_entropy(problem: _Problem, labels, upstream, *, detached: bool, logits_dtype: str) -> _Run:
    problem.reset_grads()
    hidden = problem.local_hidden()
    head = LMHeadCapture(logits_dtype=logits_dtype, tile_rows=7).mtp_output_layer(problem.layer)
    loss = head(
        hidden,
        weight=problem.layer.weight.detach() if detached else None,
        output_cross_entropy_loss=True,
        labels=labels,
    )
    (loss * upstream).sum().backward()
    return _Run(loss.detach(), None, hidden.grad, problem.layer.weight.main_grad.clone(), [loss.numel()])


def _megatron_cross_entropy(problem: _Problem, labels, upstream, *, detached: bool) -> _Run:
    """What ``process_mtp_loss`` computes without the linear cross-entropy: logits, then Megatron's loss."""
    problem.reset_grads()
    hidden = problem.local_hidden()
    logits, _ = problem.layer(hidden, weight=problem.layer.weight.detach() if detached else None)
    ignored = labels == -100
    loss = vocab_parallel_cross_entropy(
        logits, labels.masked_fill(ignored, 0).t().contiguous(), tp_group=get_parallel_state().tp.group
    )
    loss = loss.t().masked_fill(ignored, 0.0)
    (loss * upstream).sum().backward()
    return _Run(loss.detach(), None, hidden.grad, problem.layer.weight.main_grad.clone(), [loss.numel()])


# ---------------------------------------------------------------------------
# The losses: SFT, PPO and CISPO through objective.loss_function
# ---------------------------------------------------------------------------

_LOSS_CASES = {
    "sft": dict(loss_type="sft_loss", rollout_temperature=1.0),
    "ppo_entropy_regularized_kl": dict(
        entropy_coef=0.01, use_kl_loss=True, kl_loss_coef=0.1, kl_loss_type="low_var_kl"
    ),
    "ppo_entropy_diagnostic": dict(entropy_coef=0.0, observe_training_entropy=True),
    "ppo_entropy_off_recompute": dict(entropy_coef=0.0, recompute_loss_function=True),
    "cispo_rollout_logprobs": dict(
        policy_loss_type="cispo",
        eps_clip=0.2,
        eps_clip_high=0.28,
        entropy_coef=0.0,
        observe_training_entropy=True,
        use_rollout_logprobs=True,
        skip_actor_forward_only=True,
        calculate_per_token_loss=True,
    ),
    "cispo_entropy_regularized_recompute": dict(
        policy_loss_type="cispo", entropy_coef=0.01, recompute_loss_function=True
    ),
}


def _loss_batch(problem: _Problem, reference_log_probs: list[torch.Tensor]) -> dict:
    """Old, rollout and reference scores around the current ones: ratios on both sides of the clip,
    advantages of both signs, and nonfinite values only where the loss mask is zero."""
    gen = torch.Generator().manual_seed(17)
    offsets = torch.tensor([-0.6, -0.1, 0.0, 0.1, 0.6])
    batch = dict(
        unconcat_tokens=problem.tokens,
        total_lengths=problem.total_lens,
        response_lengths=problem.response_lens,
        loss_masks=[],
        log_probs=[],
        rollout_log_probs=[],
        ref_log_probs=[],
        advantages=[],
    )
    if problem.max_seq_lens is not None:
        batch["max_seq_lens"] = problem.max_seq_lens
    for current in reference_log_probs:
        n = current.numel()
        shift = offsets[torch.arange(n) % len(offsets)].cuda()
        mask = torch.ones(n, device="cuda")
        mask[2::7] = 0.0
        old = current + shift
        rollout = current - shift
        advantages = torch.randn(n, generator=gen).cuda()
        old[mask == 0], rollout[mask == 0], advantages[mask == 0] = float("nan"), float("-inf"), float("inf")
        batch["loss_masks"].append(mask)
        batch["log_probs"].append(old)
        batch["rollout_log_probs"].append(rollout)
        batch["ref_log_probs"].append(current + 0.05 * torch.randn(n, generator=gen).cuda())
        batch["advantages"].append(advantages)
    return batch


def _loss_run(problem: _Problem, backend: str, batch: dict, overrides: dict):
    problem.reset_grads()
    hidden = problem.local_hidden()
    args = _args(backend, log_probs_chunk_size=5, qkv_format=problem.qkv_format, **overrides)
    # scaled as the Megatron training step scales it, for one microbatch of this batch's rollouts
    loss, _, log = loss_function(
        args,
        batch,
        1,
        _model_output(problem, backend, hidden),
        apply_megatron_loss_scaling=True,
        num_rollouts=len(problem.response_lens),
    )
    loss.backward()
    metrics = dict(zip(log["keys"], log["values"][1:].tolist(), strict=True))
    return loss.detach(), metrics, hidden.grad, problem.layer.weight.main_grad.clone()


def _check_losses(*, sequence_parallel: bool) -> None:
    for qkv_format in ("thd", "bshd"):
        problem = _exact_problem(sequence_parallel, qkv_format=qkv_format)
        with torch.no_grad():
            args = _args("fused", qkv_format=qkv_format)
            current, _ = _score(problem, args, _model_output(problem, "fused", problem.local_hidden()))
        for name, overrides in _LOSS_CASES.items():
            batch = _loss_batch(problem, current)
            behavior = {key: [t.clone() for t in batch[key]] for key in ("log_probs", "rollout_log_probs")}
            loss, metrics, hidden_grad, main_grad = _loss_run(problem, "fused", batch, overrides)
            lm_loss, lm_metrics, lm_hidden_grad, lm_main_grad = _loss_run(problem, "fused_lm_head", batch, overrides)
            assert torch.isfinite(lm_loss), name
            torch.testing.assert_close(lm_loss, loss, rtol=0, atol=0, msg=name)
            assert lm_metrics.keys() == metrics.keys(), name
            for key, value in metrics.items():
                assert lm_metrics[key] == pytest.approx(value, rel=1e-6, abs=1e-7, nan_ok=True), (name, key)
            for got, want in ((lm_hidden_grad, hidden_grad), (lm_main_grad, main_grad)):
                assert _relative_error(got, want) < 1e-2, name
            # the trainer never overwrites the behavior policy's scores
            for key, tensors in behavior.items():
                for before, after in zip(tensors, batch[key], strict=True):
                    torch.testing.assert_close(after, before, rtol=0, atol=0, equal_nan=True)


# ---------------------------------------------------------------------------
# Memory: the vocabulary-sized workspace is one tile, not the scored rows
# ---------------------------------------------------------------------------

_MEMORY_VOCAB = 131_072
_MEMORY_HIDDEN = 256
_MEMORY_ROWS = 8192


def _check_memory_is_tile_bounded(*, sequence_parallel: bool) -> None:
    del sequence_parallel
    gen = torch.Generator().manual_seed(3)
    full_weight = (torch.randn(_MEMORY_VOCAB, _MEMORY_HIDDEN, generator=gen) * 0.05).to(torch.bfloat16).cuda()
    layer = _layer(False, full_weight)
    tokens = [torch.randint(0, _MEMORY_VOCAB, (_MEMORY_ROWS + 1,), generator=gen).cuda()]
    hidden_values = torch.randn(_MEMORY_ROWS + 1, 1, _MEMORY_HIDDEN, generator=gen).to(torch.bfloat16).cuda()
    materialized_bytes = (_MEMORY_ROWS + 1) * _MEMORY_VOCAB * 2
    hidden_bytes = _MEMORY_ROWS * _MEMORY_HIDDEN * 2
    weight_bytes = full_weight.numel() * 2  # the dummy wgrad Megatron hands autograd
    peaks = {}
    for logits_dtype, element_bytes in (("model", 2), ("fp32", 4)):
        for tile_rows in (512, 2048):
            # the first step allocates what persists for the run, such as cuBLAS's workspace
            _memory_step(layer, hidden_values, tokens, logits_dtype, tile_rows)
            retained, forward_peak, backward_peak = _memory_step(layer, hidden_values, tokens, logits_dtype, tile_rows)
            tile_bytes = tile_rows * _MEMORY_VOCAB * element_bytes
            where = (logits_dtype, tile_rows, retained, forward_peak, backward_peak)
            # across tiles the forward keeps the scored hidden rows and per-row fp32 statistics
            assert retained < 2 * hidden_bytes + 64 * _MEMORY_ROWS, where
            assert forward_peak < 2.5 * tile_bytes + 2 * hidden_bytes, where
            assert backward_peak < 3 * tile_bytes + 4 * hidden_bytes + weight_bytes, where
            if tile_rows * 16 <= _MEMORY_ROWS:
                assert backward_peak < materialized_bytes / 2, where
            peaks[logits_dtype, tile_rows] = (forward_peak, backward_peak)
    for logits_dtype in ("model", "fp32"):
        assert peaks[logits_dtype, 2048][0] > peaks[logits_dtype, 512][0], peaks
    if dist.get_rank() == 0:
        print({key: tuple(f"{value / 2**20:.0f} MiB" for value in pair) for key, pair in peaks.items()})


def _memory_step(layer, hidden_values, tokens, logits_dtype, tile_rows) -> tuple[int, int, int]:
    """Bytes allocated above the inputs: kept after the forward, and the forward's and backward's peaks."""
    layer.weight.main_grad.zero_()
    layer.weight.grad_added_to_main_grad = False
    hidden = hidden_values.clone().requires_grad_(True)
    capture = LMHeadCapture(logits_dtype=logits_dtype, tile_rows=tile_rows)
    capture.record(layer, None, hidden)
    args = _args("fused_lm_head", log_probs_chunk_size=tile_rows, vocab_size=_MEMORY_VOCAB, rollout_temperature=1.0)
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    res = get_log_probs_and_entropy(
        capture.inputs(hidden),
        args=args,
        unconcat_tokens=tokens,
        total_lengths=[_MEMORY_ROWS + 1],
        response_lengths=[_MEMORY_ROWS],
        with_entropy=True,
    )
    loss = torch.cat(res["log_probs"]).sum() + 0.1 * torch.cat(res["entropy"]).sum()
    torch.cuda.synchronize()
    retained = torch.cuda.memory_allocated() - base
    forward_peak = torch.cuda.max_memory_allocated() - base
    torch.cuda.reset_peak_memory_stats()
    loss.backward()
    torch.cuda.synchronize()
    return retained, forward_peak, torch.cuda.max_memory_allocated() - base


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


_CHECKS = (
    _check_parity,
    _check_entropy_modes,
    _check_fp32_logits,
    _check_no_grad_scores,
    _check_empty_responses,
    _check_losses,
    _check_frozen_weight,
    _check_weight_grad_hook,
    _check_mtp_cross_entropy,
    _check_rejections,
)
_SINGLE_RANK_CHECKS = (_check_zigzag_cp_ranks, _check_memory_is_tile_bounded)


@pytest.mark.parametrize(("tp", "sequence_parallel"), _LAYOUTS, ids=_LAYOUT_IDS)
def test_fused_lm_head(tp, sequence_parallel):
    """Every check in one world per layout: starting a world costs more than most checks."""
    _spawn(tp, sequence_parallel, _CHECKS + (_SINGLE_RANK_CHECKS if tp == 1 else ()))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
