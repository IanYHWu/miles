from model_args_utils import load_sibling_model_args


def model_args() -> str:
    return load_sibling_model_args(
        __file__, "qwen3.5-35B-A3B", nlayers=48, hidden_size=3072, attention_heads=32, expert_hidden_size=1024
    )
