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
    Load GPT-2 or GPT-Neo model with TransformerLens. (medium, large, or gpt-neo-125M)
    """
    if model_name not in ["gpt2-medium", "gpt2-large", "gpt-neo-125M"]:
        raise ValueError(f"Unsupported model: {model_name}. Must be 'gpt2-medium', 'gpt2-large', or 'gpt-neo-125M'")
    
    if model_name == "gpt-neo-125M":
        model_name = "EleutherAI/gpt-neo-125M"
    
    model = HookedTransformer.from_pretrained(pretrained_name)
    model.eval()
    
    print(f"Model loaded: {model_name}")
    print(f"  Layers: {model.cfg.n_layers}")
    print(f"  Hidden size: {model.cfg.d_model}")
    print(f"  Attention heads: {model.cfg.n_heads}")
    
    return model

def get_tokenizer(model_name: str = "gpt2-medium") -> GPT2Tokenizer:
    """Load GPT-2 or GPT-Neo tokenizer. (medium, large, or gpt-neo-125M)"""
    if model_name not in ["gpt2-medium", "gpt2-large", "gpt-neo-125M"]:
        raise ValueError(f"Unsupported model: {model_name}. Must be 'gpt2-medium', 'gpt2-large', or 'gpt-neo-125M'")
    
    if model_name == "gpt-neo-125M":
        tokenizer_name = "EleutherAI/gpt-neo-125M"
    else:
        tokenizer_name = model_name
    
    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def setup_device() -> torch.device:
    """
    Setup compute device (CUDA if available, else CPU).
    
    Returns:
        torch.device
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

