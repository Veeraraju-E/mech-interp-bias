"""Model setup for GPT-2 and GPT-Neo using TransformerLens."""

import torch
from transformer_lens import HookedTransformer
from transformers import GPT2Tokenizer, GPTNeoXTokenizerFast, AutoTokenizer


# Model name mapping for TransformerLens
MODEL_NAME_MAP = {
    "gpt2": "gpt2",
    "gpt2-medium": "gpt2-medium",
    "gpt2-large": "gpt2-large",
    "gpt2-xl": "gpt2-xl",
    "gpt-neo-125m": "EleutherAI/gpt-neo-125M",
    "gpt-neo-1.3b": "EleutherAI/gpt-neo-1.3B",
    "gpt-neo-2.7b": "EleutherAI/gpt-neo-2.7B",
}


def load_model(model_name: str = "gpt2-medium") -> HookedTransformer:
    """
    Load language model with TransformerLens.
    
    Args:
        model_name: Model identifier (gpt2-medium, gpt-neo-125m, gpt-neo-1.3b, gpt-neo-2.7b)
    
    Returns:
        HookedTransformer instance
    """
    if model_name not in MODEL_NAME_MAP:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Supported models: {list(MODEL_NAME_MAP.keys())}"
        )
    
    pretrained_name = MODEL_NAME_MAP[model_name]
    print(f"Loading model: {pretrained_name}")
    
    model = HookedTransformer.from_pretrained(pretrained_name)
    model.eval()
    
    print(f"Model loaded: {model_name}")
    print(f"  Layers: {model.cfg.n_layers}")
    print(f"  Hidden size: {model.cfg.d_model}")
    print(f"  Attention heads: {model.cfg.n_heads}")
    
    return model


def get_tokenizer(model_name: str = "gpt2-medium"):
    """
    Load tokenizer for the specified model.
    
    Args:
        model_name: Model identifier
    
    Returns:
        Tokenizer instance
    """
    if model_name not in MODEL_NAME_MAP:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Supported models: {list(MODEL_NAME_MAP.keys())}"
        )
    
    pretrained_name = MODEL_NAME_MAP[model_name]
    
    # Use AutoTokenizer for better compatibility
    tokenizer = AutoTokenizer.from_pretrained(pretrained_name)
    
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return tokenizer


def setup_device() -> torch.device:
    """
    Setup compute device (CUDA if available, else CPU).
    
    Returns:
        torch.device
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

