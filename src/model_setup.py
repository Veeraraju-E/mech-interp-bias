"""Model setup for GPT-2 Medium using TransformerLens."""

import torch
from transformer_lens import HookedTransformer
from transformers import GPT2Tokenizer


def load_model() -> HookedTransformer:
    """
    Load GPT-2 Medium model with TransformerLens.
    
    Returns:
        HookedTransformer instance for GPT-2 Medium
    """
    model = HookedTransformer.from_pretrained("gpt2-medium")
    model.eval()
    return model


def get_tokenizer() -> GPT2Tokenizer:
    """
    Load GPT-2 tokenizer.
    
    Returns:
        GPT2Tokenizer instance
    """
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def setup_device() -> torch.device:
    """
    Setup compute device (CUDA if available, else CPU).
    
    Returns:
        torch.device
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

