"""Head and MLP ablation functions for bias analysis."""

import torch
from typing import Dict, List, Callable, Tuple
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint


def ablate_head(
    model: HookedTransformer,
    layer: int,
    head: int,
    examples: List[torch.Tensor],
    bias_metric_fn: Callable[[torch.Tensor], float]
) -> float:
    """
    Ablate a specific attention head and measure bias change.
    
    Args:
        model: HookedTransformer model
        layer: Layer index
        head: Head index within layer
        examples: List of tokenized input prompts
        bias_metric_fn: Function that computes bias from logits
    
    Returns:
        Change in bias metric: bias(ablated) - bias(original)
    """
    tokens = examples[0].unsqueeze(0).to(model.cfg.device)
    
    # Get original bias
    with torch.no_grad():
        original_logits = model(tokens)
        original_bias = bias_metric_fn(original_logits)
    
    # Ablate head by zeroing its output
    hook_name = f"blocks.{layer}.attn.hook_z"
    
    def zero_head_hook(activation: torch.Tensor, hook: HookPoint):
        # Zero out the specific head
        # activation shape: [batch, seq, n_heads, head_dim]
        activation[:, :, head, :] = 0.0
        return activation
    
    hooks = [(hook_name, zero_head_hook)]
    
    with torch.no_grad():
        ablated_logits = model.run_with_hooks(tokens, fwd_hooks=hooks)
        ablated_bias = bias_metric_fn(ablated_logits)
    
    return ablated_bias - original_bias


def ablate_mlp(
    model: HookedTransformer,
    layer: int,
    examples: List[torch.Tensor],
    bias_metric_fn: Callable[[torch.Tensor], float]
) -> float:
    """
    Ablate MLP in a specific layer and measure bias change.
    
    Args:
        model: HookedTransformer model
        layer: Layer index
        examples: List of tokenized input prompts
        bias_metric_fn: Function that computes bias from logits
    
    Returns:
        Change in bias metric: bias(ablated) - bias(original)
    """
    tokens = examples[0].unsqueeze(0).to(model.cfg.device)
    
    # Get original bias
    with torch.no_grad():
        original_logits = model(tokens)
        original_bias = bias_metric_fn(original_logits)
    
    # Ablate MLP by zeroing its output
    hook_name = f"blocks.{layer}.hook_mlp_out"
    
    def zero_mlp_hook(activation: torch.Tensor, hook: HookPoint):
        return torch.zeros_like(activation)
    
    hooks = [(hook_name, zero_mlp_hook)]
    
    with torch.no_grad():
        ablated_logits = model.run_with_hooks(tokens, fwd_hooks=hooks)
        ablated_bias = bias_metric_fn(ablated_logits)
    
    return ablated_bias - original_bias


def scan_all_heads(
    model: HookedTransformer,
    examples: List[torch.Tensor],
    bias_metric_fn: Callable[[torch.Tensor], float]
) -> Dict[Tuple[int, int], float]:
    """
    Scan all attention heads and compute their impact on bias.
    
    Args:
        model: HookedTransformer model
        examples: List of tokenized input prompts
        bias_metric_fn: Function that computes bias from logits
    
    Returns:
        Dictionary mapping (layer, head) tuples to bias change
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    
    impact_scores = {}
    
    for layer in range(n_layers):
        for head in range(n_heads):
            try:
                impact = ablate_head(model, layer, head, examples, bias_metric_fn)
                impact_scores[(layer, head)] = impact
            except Exception:
                impact_scores[(layer, head)] = 0.0
    
    return impact_scores


def scan_all_mlps(
    model: HookedTransformer,
    examples: List[torch.Tensor],
    bias_metric_fn: Callable[[torch.Tensor], float]
) -> Dict[int, float]:
    """
    Scan all MLPs and compute their impact on bias.
    
    Args:
        model: HookedTransformer model
        examples: List of tokenized input prompts
        bias_metric_fn: Function that computes bias from logits
    
    Returns:
        Dictionary mapping layer index to bias change
    """
    n_layers = model.cfg.n_layers
    
    impact_scores = {}
    
    for layer in range(n_layers):
        try:
            impact = ablate_mlp(model, layer, examples, bias_metric_fn)
            impact_scores[layer] = impact
        except Exception:
            impact_scores[layer] = 0.0
    
    return impact_scores

