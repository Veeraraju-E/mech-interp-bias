"""Head and MLP ablation functions for bias analysis."""

import torch
from typing import Dict, List, Callable, Tuple
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint
from tqdm import tqdm


def ablate_head(
    model: HookedTransformer,
    layer: int,
    head: int,
    examples: List[torch.Tensor],
    bias_metric_fn: Callable[[torch.Tensor], float]
) -> float:
    """Ablate a specific attention head and measure bias change (averaged over examples)."""
    hook_name = f"blocks.{layer}.attn.hook_z"
    
    def zero_head_hook(activation: torch.Tensor, hook: HookPoint):
        activation = activation.clone()
        activation[:, :, head, :] = 0.0
        return activation
    
    hooks = [(hook_name, zero_head_hook)]
    bias_changes = []
    
    for tokens in examples:
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
        tokens = tokens.to(model.cfg.device)
        
        with torch.no_grad():
            original_logits = model(tokens)
            original_bias = bias_metric_fn(original_logits)
            ablated_logits = model.run_with_hooks(tokens, fwd_hooks=hooks)
            ablated_bias = bias_metric_fn(ablated_logits)
        
        bias_changes.append(ablated_bias - original_bias)
    
    return sum(bias_changes) / len(bias_changes) if bias_changes else 0.0


def ablate_mlp(
    model: HookedTransformer,
    layer: int,
    examples: List[torch.Tensor],
    bias_metric_fn: Callable[[torch.Tensor], float]
) -> float:
    """Ablate MLP in a specific layer and measure bias change (averaged over examples)."""
    hook_name = f"blocks.{layer}.hook_mlp_out"
    
    def zero_mlp_hook(activation: torch.Tensor, hook: HookPoint):
        return torch.zeros_like(activation)
    
    hooks = [(hook_name, zero_mlp_hook)]
    bias_changes = []
    
    for tokens in examples:
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
        tokens = tokens.to(model.cfg.device)
        
        with torch.no_grad():
            original_logits = model(tokens)
            original_bias = bias_metric_fn(original_logits)
            ablated_logits = model.run_with_hooks(tokens, fwd_hooks=hooks)
            ablated_bias = bias_metric_fn(ablated_logits)
        
        bias_changes.append(ablated_bias - original_bias)
    
    return sum(bias_changes) / len(bias_changes) if bias_changes else 0.0


def scan_all_heads(
    model: HookedTransformer,
    examples: List[torch.Tensor],
    bias_metric_fn: Callable[[torch.Tensor], float]
) -> Dict[Tuple[int, int], float]:
    """Scan all attention heads and compute their impact on bias."""
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    impact_scores = {}
    
    total_heads = n_layers * n_heads
    with tqdm(total=total_heads, desc="Scanning heads") as pbar:
        for layer in range(n_layers):
            for head in range(n_heads):
                try:
                    impact = ablate_head(model, layer, head, examples, bias_metric_fn)
                    impact_scores[(layer, head)] = impact
                except Exception:
                    impact_scores[(layer, head)] = 0.0
                pbar.update(1)
    
    return impact_scores


def scan_all_mlps(
    model: HookedTransformer,
    examples: List[torch.Tensor],
    bias_metric_fn: Callable[[torch.Tensor], float]
) -> Dict[int, float]:
    """Scan all MLPs and compute their impact on bias."""
    n_layers = model.cfg.n_layers
    impact_scores = {}
    
    for layer in tqdm(range(n_layers), desc="Scanning MLPs"):
        try:
            impact = ablate_mlp(model, layer, examples, bias_metric_fn)
            impact_scores[layer] = impact
        except Exception:
            impact_scores[layer] = 0.0
    
    return impact_scores

