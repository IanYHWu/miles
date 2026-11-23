"""Push trainer weights into externally run vLLM servers over NCCL.

miles' own weight sync (``WeightUpdater`` and its protocols) targets the SGLang engines the
inference controller owns. This package is the trainer side of vLLM's RLHF weight-transfer API
for servers started outside miles (``vllm serve --weight-transfer-config '{"backend": "nccl"}'``
with ``VLLM_SERVER_DEV_MODE=1``), so the miles image needs no vLLM:

* ``nccl``: the few NCCL entry points the sender needs, loaded with ctypes from the serving
  image's libnccl (NCCL refuses communicators whose ranks run different versions).
* ``pusher``: global trainer rank 0's sender, one communicator per server, so a server that dies
  mid-push only drops its own link.
* ``updater``: the collective entry point every trainer rank calls; rank 0 sends while the other
  ranks join the Megatron->HF iterator's gathers.

See docs/advanced/vllm-weight-push.md.
"""
