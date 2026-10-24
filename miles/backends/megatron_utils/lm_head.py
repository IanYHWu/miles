"""Megatron side of ``--log-probs-backend fused_lm_head``: the GPT model returns its LM head's input.

``GPTModel.forward(output_processor=...)`` hands the last pipeline stage's final hidden states and its
output layer to the processor in place of the projection. A fresh ``LMHeadCapture`` per forward call
records that layer's projection, so concurrently live microbatches never share one; the loss or scoring
callback then receives the hidden states as ``LMHeadInputs`` and projects only the rows it scores
(``loss.hub.tiled_lm_head``).

Under ``--enable-mtp-training`` Megatron computes the MTP loss itself, before the output processor, by
projecting each MTP depth to full logits. ``install_mtp_loss`` routes the forwards that run under
``LMHeadCapture.active`` through Megatron's linear cross-entropy branch of that loss instead, with the
tiled head as the output layer; Megatron keeps the label shifts, masking, scaling and logging.

``--lm-head-logits-dtype model`` projects through the output layer's own Megatron linear, as the
logits path does. ``fp32`` keeps bf16 operands and accumulates and returns the logits in fp32, so the
softmax never sees them rounded; its backward is Megatron's on the logits gradient rounded once to bf16.
"""

import contextlib
import contextvars
import functools
import inspect
import logging
from argparse import Namespace
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
from megatron.core.models.gpt import gpt_model as megatron_gpt_model
from megatron.core.tensor_parallel import layers as megatron_layers
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, LinearWithGradAccumulationAndAsyncCommunication
from megatron.core.tensor_parallel.mappings import (
    copy_to_tensor_model_parallel_region,
    gather_from_sequence_parallel_region,
)
from megatron.core.transformer import multi_token_prediction as megatron_mtp
from megatron.core.utils import unwrap_model

from miles.backends.training_utils.loss.hub.lm_head import LMHeadInputs, LMHeadLogitsDtype
from miles.backends.training_utils.parallel import get_parallel_state

logger = logging.getLogger(__name__)

# the losses that score the policy's tokens; every other loss keeps the logits path
LM_HEAD_LOSS_TYPES = ("policy_loss", "sft_loss")


@dataclass(frozen=True)
class MegatronLMHeadProjection:
    """One forward call's output layer, its runtime weight and its parallel flags, frozen at capture.

    Freezing ``sequence_parallel`` here keeps a later toggle of the shared layer from changing how a
    checkpointed tile replays.
    """

    output_layer: ColumnParallelLinear
    weight: torch.Tensor  # the layer's weight, the tied embedding weight, or either detached (MTP heads)
    tp_group: dist.ProcessGroup
    sequence_parallel: bool
    allreduce_dgrad: bool
    gradient_accumulation_fusion: bool
    logits_dtype: LMHeadLogitsDtype

    def gather_sequence(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """The local sequence's ``[S, B, H]`` hidden states; the backward sums their gradient over TP.

        This is the input communication the output layer's linear would do, done once per microbatch so
        the tiles' linears need none.
        """
        if self.sequence_parallel:
            return gather_from_sequence_parallel_region(
                hidden_states, tensor_parallel_output_grad=True, group=self.tp_group
            )
        if self.allreduce_dgrad:
            return copy_to_tensor_model_parallel_region(hidden_states, group=self.tp_group)
        return hidden_states

    def track_weight_grad(self, hidden_rows: torch.Tensor) -> torch.Tensor:
        """``hidden_rows`` for the tiles to project, with the weight's DDP bookkeeping after their backward."""
        weight = self.weight
        if not (
            torch.is_grad_enabled()
            and weight.requires_grad
            and self.gradient_accumulation_fusion
            and hasattr(weight, "grad_added_to_main_grad")
        ):
            return hidden_rows
        return _WeightGradReady.apply(hidden_rows, weight)

    def project(self, hidden_rows: torch.Tensor) -> torch.Tensor:
        """``[rows, V / TP]`` logits; a trainable weight's gradient accumulates as the output layer's does."""
        if self.logits_dtype == "fp32":
            return _Fp32LogitsLinear.apply(hidden_rows, self.weight, self.gradient_accumulation_fusion, self.tp_group)
        # Megatron's linear, or its frozen-weight one for a weight that takes no gradient
        return self.output_layer._forward_impl(
            input=hidden_rows,
            weight=self.weight,
            bias=None,
            gradient_accumulation_fusion=self.gradient_accumulation_fusion,
            allreduce_dgrad=False,  # gather_sequence reduces the input gradient
            sequence_parallel=False,  # the rows are already gathered
            grad_output_buffer=None,
            wgrad_deferral_limit=None,
            tp_group=self.tp_group,
        )


def _projection(output_layer, weight: torch.Tensor, logits_dtype: LMHeadLogitsDtype) -> MegatronLMHeadProjection:
    return MegatronLMHeadProjection(
        output_layer=output_layer,
        weight=weight,
        tp_group=output_layer.tp_group,
        sequence_parallel=output_layer.sequence_parallel,
        allreduce_dgrad=output_layer.allreduce_dgrad,
        gradient_accumulation_fusion=output_layer.gradient_accumulation_fusion,
        logits_dtype=logits_dtype,
    )


class _WeightGradReady(torch.autograd.Function):
    """Identity on the rows the tiles project that finishes Megatron's weight-gradient bookkeeping.

    Megatron's linear accumulates the weight gradient into ``main_grad``, then sets
    ``grad_added_to_main_grad`` and hands autograd a dummy weight gradient, so the weight's DDP hook
    fires; under ``--overlap-grad-reduce`` its bucket waits for that hook. Inside a tile's checkpoint the
    backward reads the replayed weight, a detached tensor without those attributes, and does neither.
    The tiles consume this node's output, so its backward runs after every tile's and does both for the
    parameter itself.
    """

    @staticmethod
    def forward(ctx, hidden_rows, weight):
        ctx.weight = weight  # the parameter itself: saved for backward, it would come back detached
        return hidden_rows.view_as(hidden_rows)

    @staticmethod
    def backward(ctx, grad_rows):
        weight = ctx.weight
        weight.grad_added_to_main_grad = True
        return grad_rows, _dummy_wgrad(weight, grad_rows.dtype)


def _dummy_wgrad(weight: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """The placeholder weight gradient Megatron's linear returns once ``main_grad`` holds the real one."""
    zero = getattr(weight, "zero_out_wgrad", False)  # a weight that also embeds tokens on this stage
    shape = list(weight.main_grad.shape)
    if megatron_layers.HAVE_TE:
        return megatron_layers.get_dummy_wgrad(shape, dtype, zero=zero)
    return (torch.zeros if zero else torch.empty)(shape, dtype=dtype, device=weight.main_grad.device)


class _Fp32LogitsLinear(LinearWithGradAccumulationAndAsyncCommunication):
    """Megatron's linear on an unsharded input without bias, with logits accumulated and returned in fp32.

    The forward sets the context the parent's backward reads; the backward is the parent's, with its
    ``main_grad`` accumulation, run on the fp32 logits gradient rounded once to the operand dtype. A
    weight that takes no gradient gets only the input gradient, as Megatron's frozen-weight linear.
    """

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx, input, weight, gradient_accumulation_fusion, tp_group):
        ctx.weight_requires_grad = weight.requires_grad
        ctx.save_for_backward(input if weight.requires_grad else None, weight)
        ctx.main_grad = weight.main_grad if gradient_accumulation_fusion and hasattr(weight, "main_grad") else None
        ctx.use_bias = False
        ctx.gradient_accumulation_fusion = gradient_accumulation_fusion
        ctx.allreduce_dgrad = False
        ctx.sequence_parallel = False
        ctx.wgrad_deferral_limit = None
        ctx.grad_output_buffer = None
        ctx.tp_group = tp_group
        ctx.operand_dtype = input.dtype
        return torch.mm(input, weight.t(), out_dtype=torch.float32)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.to(ctx.operand_dtype)
        if not ctx.weight_requires_grad:
            _, weight = ctx.saved_tensors
            return grad_output.matmul(weight), None, None, None
        grads = LinearWithGradAccumulationAndAsyncCommunication.backward(ctx, grad_output)
        grad_input, grad_weight = grads[:2]
        return grad_input, grad_weight, None, None


# the capture of the forward call now running, for the MTP loss Megatron computes inside it
_ACTIVE_CAPTURE: contextvars.ContextVar["LMHeadCapture | None"] = contextvars.ContextVar(
    "miles_lm_head_capture", default=None
)


class LMHeadCapture:
    """One forward call's LM head, recorded by its ``output_processor`` on the last pipeline stage."""

    def __init__(self, *, logits_dtype: LMHeadLogitsDtype, tile_rows: int) -> None:
        self._logits_dtype = logits_dtype
        self._tile_rows = tile_rows
        self._projection: MegatronLMHeadProjection | None = None

    def forward_kwargs(self) -> dict[str, Any]:
        return dict(output_processor=_capture_lm_head, output_processor_context=self)

    @contextlib.contextmanager
    def active(self) -> Iterator[None]:
        """Run the model's forward under this capture, so its MTP loss goes through the tiles too."""
        token = _ACTIVE_CAPTURE.set(self)
        try:
            yield
        finally:
            _ACTIVE_CAPTURE.reset(token)

    def bind(self, callback: Callable[..., Any]) -> Callable[..., Any]:
        """``callback`` with this forward's ``LMHeadInputs`` where Megatron passes the output tensor."""
        return functools.partial(_call_with_lm_head_inputs, callback, self)

    def record(self, output_layer, output_weight: torch.Tensor | None, hidden_states: torch.Tensor) -> None:
        _check_output_layer(output_layer)
        weight = output_layer.weight if output_weight is None else output_weight
        _check_operand_dtypes(hidden_states, weight)
        self._projection = _projection(output_layer, weight, self._logits_dtype)

    def inputs(self, hidden_states: torch.Tensor) -> LMHeadInputs:
        if self._projection is None:
            raise RuntimeError(
                "--log-probs-backend fused_lm_head: the model returned without calling its output_processor, "
                "so its output is not the LM head's input"
            )
        return LMHeadInputs(hidden_states=hidden_states, projection=self._projection)

    def mtp_output_layer(self, output_layer) -> "_TiledLinearCrossEntropy":
        _check_output_layer(output_layer)
        return _TiledLinearCrossEntropy(output_layer, logits_dtype=self._logits_dtype, tile_rows=self._tile_rows)


def lm_head_capture(args: Namespace, *, scores_log_probs: bool) -> LMHeadCapture | None:
    """A fresh capture for one forward call whose log-probs come from the bounded LM head, else ``None``."""
    if args.log_probs_backend != "fused_lm_head" or not scores_log_probs:
        return None
    return LMHeadCapture(logits_dtype=args.lm_head_logits_dtype, tile_rows=args.log_probs_chunk_size)


def _capture_lm_head(*, hidden_states, output_layer, output_weight, context: LMHeadCapture, **_unused):
    context.record(output_layer, output_weight, hidden_states)
    return hidden_states


def _call_with_lm_head_inputs(callback, capture: LMHeadCapture, output_tensor, *args, **kwargs):
    return callback(capture.inputs(output_tensor), *args, **kwargs)


def _check_output_layer(output_layer) -> None:
    if not isinstance(output_layer, ColumnParallelLinear):
        raise ValueError(
            f"--log-probs-backend fused_lm_head projects through Megatron's ColumnParallelLinear, "
            f"not {type(output_layer).__name__}"
        )
    if output_layer.bias is not None or output_layer.explicit_expert_comm or output_layer.disable_grad_reduce:
        raise ValueError("--log-probs-backend fused_lm_head supports a bias-free, TP-reduced output layer only")


def _check_operand_dtypes(hidden_states: torch.Tensor, weight: torch.Tensor) -> None:
    if hidden_states.dtype != weight.dtype:
        raise ValueError(
            f"--log-probs-backend fused_lm_head projects {hidden_states.dtype} hidden states with a "
            f"{weight.dtype} weight; the forward must not upcast its output (fp32_output=False)"
        )


# ---------------------------------------------------------------------------
# The MTP loss
# ---------------------------------------------------------------------------


class _TiledLinearCrossEntropy:
    """An output layer as Megatron's linear cross-entropy contract calls it, scored tile by tile.

    ``process_mtp_loss`` calls ``output_layer(hidden, weight=..., runtime_gather_output=...,
    output_cross_entropy_loss=True, labels=...)`` for the per-token loss, and reads ``weight`` to detach it
    under ``--mtp-detach-heads``.
    """

    def __init__(self, output_layer: ColumnParallelLinear, *, logits_dtype: LMHeadLogitsDtype, tile_rows: int):
        self._output_layer = output_layer
        self._logits_dtype = logits_dtype
        self._tile_rows = tile_rows

    @property
    def weight(self) -> torch.Tensor:
        return self._output_layer.weight

    def __call__(
        self,
        input_: torch.Tensor,
        weight: torch.Tensor | None = None,
        runtime_gather_output: bool | None = None,
        output_cross_entropy_loss: bool = False,
        labels: torch.Tensor | None = None,
        reduction: str = "none",
        ignore_index: int = -100,
    ) -> torch.Tensor:
        del runtime_gather_output  # the loss is per token; no logits leave the tiles
        if not output_cross_entropy_loss or labels is None or reduction != "none":
            raise ValueError("the tiled LM head only returns the per-token cross-entropy of labels")
        # the op needs triton, which not every host that imports the Megatron backend has
        from miles.backends.training_utils.loss.hub.tiled_lm_head import tiled_cross_entropy

        weight = self._output_layer.weight if weight is None else weight
        _check_operand_dtypes(input_, weight)
        return tiled_cross_entropy(
            input_,
            labels,
            projection=_projection(self._output_layer, weight, self._logits_dtype),
            tile_rows=self._tile_rows,
            ignore_index=ignore_index,
        )


class _LinearCrossEntropyConfig:
    """A model config as ``process_mtp_loss`` reads it, with the output layer's linear cross-entropy chosen."""

    cross_entropy_loss_fusion = True
    cross_entropy_fusion_impl = "linear"

    def __init__(self, config) -> None:
        self._config = config

    def __getattr__(self, name: str):
        return getattr(self._config, name)


def _process_mtp_loss(**kwargs) -> torch.Tensor:
    """Megatron's MTP loss; under an active capture, through its linear cross-entropy branch on the tiles."""
    capture = _ACTIVE_CAPTURE.get()
    if capture is not None:
        kwargs.update(
            output_layer=capture.mtp_output_layer(kwargs["output_layer"]),
            config=_LinearCrossEntropyConfig(kwargs["config"]),
        )
    return megatron_mtp.process_mtp_loss(**kwargs)


def install_mtp_loss() -> None:
    """Point ``GPTModel``'s MTP loss at ``_process_mtp_loss``; forwards without a capture keep Megatron's."""
    if megatron_gpt_model.process_mtp_loss is _process_mtp_loss:
        return
    parameters = inspect.signature(megatron_mtp.process_mtp_loss).parameters
    if megatron_gpt_model.process_mtp_loss is not megatron_mtp.process_mtp_loss or not {
        "output_layer",
        "config",
    } <= set(parameters):
        raise ValueError(
            "--log-probs-backend fused_lm_head cannot route this Megatron's MTP loss through the tiled head: "
            "GPTModel no longer computes it with multi_token_prediction.process_mtp_loss(output_layer, config)"
        )
    megatron_gpt_model.process_mtp_loss = _process_mtp_loss


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------


def setup_lm_head(args: Namespace, model: Sequence[torch.nn.Module]) -> None:
    """Fail unless every chunk holding the LM head can hand back its input; route its MTP loss; log it."""
    if args.log_probs_backend != "fused_lm_head":
        return
    for chunk in unwrap_model(list(model)):
        if not getattr(chunk, "post_process", False):
            continue
        if "output_processor" not in inspect.signature(type(chunk).forward).parameters:
            raise ValueError(
                f"--log-probs-backend fused_lm_head needs a model whose forward takes output_processor; "
                f"{type(chunk).__name__} does not"
            )
        _check_output_layer(chunk.output_layer)
        config = chunk.config
        unsupported = {
            "MuP logit scaling": config.use_mup,
            "deferred embedding wgrad": config.defer_embedding_wgrad_compute,
        }
        if names := [name for name, enabled in unsupported.items() if enabled]:
            raise ValueError(f"--log-probs-backend fused_lm_head does not support {', '.join(names)}")
        if config.mtp_num_layers:
            install_mtp_loss()
        _log_lm_head_settings(args, chunk)


def _output_weight(chunk) -> torch.Tensor:
    """The weight the chunk's forward hands its output layer: the tied embedding's, or the layer's own."""
    if chunk.share_embeddings_and_output_weights:
        return chunk.shared_embedding_or_output_weight()
    return chunk.output_layer.weight


def _log_lm_head_settings(args: Namespace, chunk) -> None:
    parallel_state = get_parallel_state()
    if parallel_state.tp.rank != 0 or parallel_state.intra_dp_cp.rank != 0:
        return
    layer = chunk.output_layer
    weight = _output_weight(chunk)
    main_grad = getattr(weight, "main_grad", None)
    matmul = torch.backends.cuda.matmul
    config = chunk.config
    logger.info(
        "LM head: --log-probs-backend fused_lm_head, %d rows per tile, %s logits (%s weight and hidden states, "
        "fp32 statistics); tied=%s, TP=%d, sequence_parallel=%s, vocab %s of %d padded; "
        "%d MTP layers, their loss tiled too (detached heads=%s); "
        "gradient_accumulation_fusion=%s, main_grad=%s; allow_bf16_reduced_precision_reduction=%s, "
        "allow_fp16_reduced_precision_reduction=%s, allow_tf32=%s",
        args.log_probs_chunk_size,
        "fp32" if args.lm_head_logits_dtype == "fp32" else weight.dtype,
        weight.dtype,
        chunk.share_embeddings_and_output_weights,
        parallel_state.tp.size,
        layer.sequence_parallel,
        args.vocab_size,
        weight.size(0) * parallel_state.tp.size,
        config.mtp_num_layers or 0,
        config.mtp_detach_heads,
        layer.gradient_accumulation_fusion,
        main_grad.dtype if main_grad is not None else None,
        matmul.allow_bf16_reduced_precision_reduction,
        matmul.allow_fp16_reduced_precision_reduction,
        matmul.allow_tf32,
    )
