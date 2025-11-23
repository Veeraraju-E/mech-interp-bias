import torch
from transformer_lens import HookedTransformer
from transformers import GPT2Tokenizer


def load_model(model_name: str = "gpt2-medium") -> HookedTransformer:
    """
    Load GPT-2 model with TransformerLens. (medium or large)
    """
    if model_name not in ["gpt2-medium", "gpt2-large"]:
        raise ValueError(f"Unsupported model: {model_name}. Must be 'gpt2-medium' or 'gpt2-large'")
    
    model = HookedTransformer.from_pretrained(model_name)
    model.eval()
    return model

def get_tokenizer(model_name: str = "gpt2-medium") -> GPT2Tokenizer:
    """Load GPT-2 tokenizer. (medium or large)"""
    if model_name not in ["gpt2-medium", "gpt2-large"]:
        raise ValueError(f"Unsupported model: {model_name}. Must be 'gpt2-medium' or 'gpt2-large'")
    
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def setup_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")