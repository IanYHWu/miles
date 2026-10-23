from model_args_utils import moe_layer_freq


def model_args(nlayers=40, hidden_size=2048, attention_heads=16, expert_hidden_size=512) -> str:
    return (
        "--spec miles_plugins.models.qwen3_5 get_qwen3_5_spec "
        "--disable-bias-linear "
        "--qk-layernorm "
        "--group-query-attention "
        f"--num-attention-heads {attention_heads} "
        "--num-query-groups 2 "
        "--kv-channels 256 "
        f"--num-layers {nlayers} "
        f"--hidden-size {hidden_size} "
        f"--ffn-hidden-size {expert_hidden_size} "
        "--normalization RMSNorm "
        "--apply-layernorm-1p "
        "--position-embedding-type rope "
        "--norm-epsilon 1e-6 "
        "--rotary-percent 0.25 "
        "--swiglu "
        "--untie-embeddings-and-output-weights "
        "--vocab-size 248320 "
        "--rotary-base 10000000 "
        # moe
        f"--moe-ffn-hidden-size {expert_hidden_size} "
        f"--moe-shared-expert-intermediate-size {expert_hidden_size} "
        "--moe-router-score-function softmax "
        "--moe-token-dispatcher-type alltoall "
        "--moe-router-topk 8 "
        f"--moe-layer-freq {moe_layer_freq(nlayers=nlayers, first_k_dense_replace=0)} "
        "--num-experts 256 "
        "--moe-grouped-gemm "
        "--moe-token-drop-policy probs "
        "--moe-router-dtype fp32 "
        "--moe-permute-fusion "
        "--moe-aux-loss-coeff 0 "
        # qwen3.5 specific
        "--attention-output-gate "
        "--moe-shared-expert-gate "
        "--mtp-num-layers 1 "
    )
