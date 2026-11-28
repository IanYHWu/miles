import asyncio
import logging
from contextlib import suppress

import uvicorn

from miles.backends.megatron_utils.megatron_config import compute_trainer_args
from miles.ray.placement_group import create_trainer_handles, push_weights
from miles.ray.rollout.router_manager import resolve_router_addrs
from miles.ray.specs.inference import compute_router_providers, create_inference_controller_handle
from miles.ray.specs.train import ACTOR_ROLE, compute_trainer_configs
from miles.ray.wiring import get_backend_capability
from miles.tinker.arguments import add_tinker_arguments, configure_tinker_args
from miles.tinker.core.service import TinkerService
from miles.tinker.core.types import GatewayConfig
from miles.tinker.runtime import FullFinetuneBackend, MilesBackend, SamplingBackend
from miles.tinker.server.app import build_app
from miles.utils.arguments import parse_args
from miles.utils.async_utils import Disposer, with_disposer
from miles.utils.hf_utils.config import load_hf_config
from miles.utils.http_utils import init_http_client
from miles.utils.orchestration_utils import init_orchestration_script

logger = logging.getLogger(__name__)


async def serve(args, *, disposer: Disposer):
    train_mode = args.tinker_train_mode
    assert train_mode != "lora" or args.multi_lora, "--tinker-train-mode lora requires --multi-lora-n-adapters > 0"
    assert args.load == args.hf_checkpoint, "Tinker trainers and engines must load the same frozen HF base"
    checkpoint_root = args.tinker_checkpoint_root or (args.save and f"{args.save}/tinker")
    assert checkpoint_root, "set --tinker-checkpoint-root (or --save to derive <save>/tinker)"
    hf_config = load_hf_config(args.hf_checkpoint).get_text_config()
    max_tokens_per_datum = hf_config.max_position_embeddings
    if args.max_tokens_per_gpu is not None:
        # The trainer pads each packed microbatch to this multiple.
        pad_size = args.tensor_model_parallel_size * args.data_pad_size_multiplier
        local_token_limit = args.max_tokens_per_gpu // pad_size * pad_size
        if args.context_parallel_size > 1:
            # each rank holds two equal zigzag chunks per datum
            local_token_limit -= local_token_limit % 2
        trainer_token_limit = local_token_limit * args.context_parallel_size
        max_tokens_per_datum = min(max_tokens_per_datum, trainer_token_limit)
    assert max_tokens_per_datum > 0, "trainer token budget must fit at least one padding block"
    _worker_manager = init_orchestration_script(args, disposer=disposer)
    init_http_client(args)

    capability = get_backend_capability(args)
    await resolve_router_addrs(args, router_providers=compute_router_providers(args, capability=capability))
    inference_controller = create_inference_controller_handle(capability=capability)
    await inference_controller.init()
    disposer.add(inference_controller)

    router_url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}"
    if train_mode == "none":
        # the engines serve the base model from the start, and no trainer stands behind them
        backend, model_config = SamplingBackend(router_url), dict(train_mode="none", n_slots=0)
    else:
        backend, model_config = await _training_backend(args, router_url, inference_controller, disposer=disposer)
    config = GatewayConfig(
        base_model=args.tinker_base_model or args.hf_checkpoint,
        checkpoint_root=checkpoint_root,
        vocab_size=hf_config.vocab_size,
        max_tokens_per_datum=max_tokens_per_datum,
        **model_config,
    )
    service = TinkerService(backend, config)

    server = uvicorn.Server(
        uvicorn.Config(
            build_app(service), host=args.tinker_server_host, port=args.tinker_server_port, log_level="info"
        )
    )
    logger.info(f"tinker gateway serving {config.base_model} on :{args.tinker_server_port}")
    # supervise both: a crashed dispatcher must take the HTTP server down with it,
    # not keep answering /healthz while every training future pends forever
    service_task = asyncio.create_task(service.run())
    server_task = asyncio.create_task(server.serve())
    try:
        done, _ = await asyncio.wait({service_task, server_task}, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            task.result()
    finally:
        for task in (service_task, server_task):
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task


async def _training_backend(args, router_url: str, inference_controller, *, disposer: Disposer) -> tuple:
    """Start the trainer and return the backend that drives it, with what its models train."""
    trainer_configs = compute_trainer_configs(args)
    [actor_config] = [config for config in trainer_configs if config.role == ACTOR_ROLE]
    trainer = create_trainer_handles(args, trainer_configs=trainer_configs)[actor_config.trainer_id]
    await trainer.init(compute_trainer_args(args, actor_config))
    disposer.add(trainer)
    actor_world_size = args.actor_num_nodes * args.actor_num_gpus_per_node
    dp_size = actor_world_size // (
        args.tensor_model_parallel_size * args.pipeline_model_parallel_size * args.context_parallel_size
    )
    if args.tinker_train_mode == "full":
        # Engines that take weight updates join the router only after a push, and the gateway serves the base model
        # from the start, before any client has saved a sampler: push the trainer's base weights once.
        await push_weights(trainer, inference_controller)
        # one client at a time trains the served model's every weight
        model_config = dict(train_mode="full", n_slots=1, trains_attn=True, trains_mlp=True, trains_unembed=True)
        return FullFinetuneBackend(trainer, router_url, inference_controller, dp_size=dp_size), model_config
    model_config = dict(
        train_mode="lora",
        n_slots=args.multi_lora_n_adapters,
        lora_alpha=args.lora_alpha,
        max_lora_rank=args.lora_rank,
        trains_attn="attn" in args.tinker_lora_groups,
        trains_mlp="mlp" in args.tinker_lora_groups,
        trains_unembed="unembed" in args.tinker_lora_groups,
    )
    return MilesBackend(trainer, router_url, dp_size=dp_size), model_config


if __name__ == "__main__":
    args = parse_args(add_tinker_arguments, entry="serve", preprocess_args=configure_tinker_args)
    # commands ship one work unit at a time; its size is the batch size
    args.use_dynamic_global_batch_size = True
    args.delay_split_train_data_by_dp = True
    asyncio.run(with_disposer(serve, args))
