import torch
from transformer_lens import HookedTransformer
from transformers import GPT2Tokenizer, AutoTokenizer
from typing import Union


# Model name mapping for TransformerLens
MODEL_NAME_MAP = {
    "gpt2": "gpt2",
    "gpt2-medium": "gpt2-medium",
    "gpt2-large": "gpt2-large",
    "gpt2-xl": "gpt2-xl",
    "gpt-neo-125m": "EleutherAI/gpt-neo-125M",
    "gpt-neo-1.3b": "EleutherAI/gpt-neo-1.3B",
    "gpt-neo-2.7b": "EleutherAI/gpt-neo-2.7B",
    "llama-3-8b": "meta-llama/Meta-Llama-3-8B",
}


def load_model(model_name: str = "gpt2-medium") -> HookedTransformer:
    if model_name not in ["gpt2-medium", "gpt2-large", "gpt-neo-125M", "llama-3-8b"]:
        raise ValueError(f"Unsupported model: {model_name}. Must be 'gpt2-medium', 'gpt2-large', 'gpt-neo-125M', or 'llama-3-8b'")
    
    if model_name == "gpt-neo-125M":
        model_name = "EleutherAI/gpt-neo-125M"
    elif model_name == "llama-3-8b":
        model_name = "meta-llama/Meta-Llama-3-8B"
    
    model = HookedTransformer.from_pretrained(model_name)
    model.eval()
    
    print(f"Model loaded: {model_name}")
    print(f"  Layers: {model.cfg.n_layers}")
    print(f"  Hidden size: {model.cfg.d_model}")
    print(f"  Attention heads: {model.cfg.n_heads}")
    
    return model

def get_tokenizer(model_name: str = "gpt2-medium") -> Union[GPT2Tokenizer, AutoTokenizer]:
    if model_name not in ["gpt2-medium", "gpt2-large", "gpt-neo-125M", "llama-3-8b"]:
        raise ValueError(f"Unsupported model: {model_name}. Must be 'gpt2-medium', 'gpt2-large', 'gpt-neo-125M', or 'llama-3-8b'")
    
    if model_name == "gpt-neo-125M":
        tokenizer_name = "EleutherAI/gpt-neo-125M"
        tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_name)
    elif model_name == "llama-3-8b":
        tokenizer_name = "meta-llama/Meta-Llama-3-8B"
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    else:
        tokenizer_name = model_name
        tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_name)
    
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def setup_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

