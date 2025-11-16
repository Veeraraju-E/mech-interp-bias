"""Causal activation patching for bias analysis.

Following methodologies from:
- Conmy et al. (2024): Edge Attribution Patching
- Chandna et al. (2025): Edge-level bias localization
- ROME: Causal tracing methodology
"""

import torch
from typing import Dict, List, Tuple, Any, Callable
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint


def get_all_hook_points(model: HookedTransformer) -> List[str]:
    """
    Get all hook points for attention heads, MLPs, and residual streams.
    
    Following Chandna et al. (2025), we patch at:
    - Attention head outputs (hook_z)
    - MLP outputs (hook_out)
    - Residual stream positions (hook_resid_post)
    
    Args:
        model: HookedTransformer model
    
    Returns:
        List of hook point names
    """
    hook_points = []
    n_layers = model.cfg.n_layers
    
    for layer in range(n_layers):
        # Attention head outputs - each head separately
        # Note: hook_z contains all heads, we'll patch individual heads via indexing
        hook_points.append(f"blocks.{layer}.attn.hook_z")
        
        # MLP outputs (correct hook name)
        hook_points.append(f"blocks.{layer}.hook_mlp_out")
        
        # Residual stream after layer
        hook_points.append(f"blocks.{layer}.hook_resid_post")
    
    return hook_points


def causal_patch_edge(
    model: HookedTransformer,
    biased_tokens: torch.Tensor,
    neutral_tokens: torch.Tensor,
    hook_name: str,
    bias_metric_fn: Callable[[torch.Tensor], float],
    context_len: int = None
) -> float:
    """
    Perform causal patching on a single edge and measure bias change.
    
    Following ROME methodology: run biased prompt and cache activations,
    then run neutral prompt and patch in biased activations one at a time.
    
    For bias analysis: we patch activations from biased run (stereotype) 
    into neutral run (antistereotype) and measure how bias changes.
    
    Args:
        model: HookedTransformer model
        biased_tokens: Tokens from biased prompt [batch, seq]
        neutral_tokens: Tokens from neutral prompt [batch, seq]
        hook_name: Hook point to patch
        bias_metric_fn: Function that computes bias from logits
        context_len: Length of context (to patch at correct position)
    
    Returns:
        Absolute change in bias metric: |L(patched) - L(original)|
    """
    # Ensure tokens are on correct device and have batch dimension
    if biased_tokens.dim() == 1:
        biased_tokens = biased_tokens.unsqueeze(0)
    if neutral_tokens.dim() == 1:
        neutral_tokens = neutral_tokens.unsqueeze(0)
    
    biased_tokens = biased_tokens.to(model.cfg.device)
    neutral_tokens = neutral_tokens.to(model.cfg.device)
    
    # Run biased prompt and cache activations
    with torch.no_grad():
        _, biased_cache = model.run_with_cache(biased_tokens)
    
    # For bias measurement, we use only the context part
    # Extract context from neutral tokens (or use full if no context_len)
    if context_len is not None and context_len <= neutral_tokens.shape[1]:
        context_tokens = neutral_tokens[:, :context_len]
    else:
        # If no context_len provided, use full neutral tokens
        context_tokens = neutral_tokens
        if context_len is None:
            context_len = neutral_tokens.shape[1]
    
    # Get original bias metric on context
    with torch.no_grad():
        original_logits = model(context_tokens)
        original_bias = bias_metric_fn(original_logits)
    
    # Patch edge: replace activation in neutral run with biased activation
    # We patch at the context end position (where we measure bias)
    # The key insight: we patch activations from the biased run (which saw the full biased sequence)
    # into the neutral run (which only saw context), at the context end position
    def patch_hook(activation: torch.Tensor, hook: HookPoint):
        # Get the corresponding activation from biased cache
        if hook_name in biased_cache:
            biased_activation = biased_cache[hook_name]
            patched_activation = activation.clone()
            
            # Patch at the context end position
            patch_pos = min(context_len - 1, activation.shape[1] - 1)
            biased_patch_pos = min(context_len - 1, biased_activation.shape[1] - 1)
            
            # Ensure positions are valid
            if (biased_activation.shape[1] > biased_patch_pos and 
                patched_activation.shape[1] > patch_pos):
                # Patch at the specific position based on activation shape
                if len(activation.shape) == 2:  # [batch, seq] - shouldn't happen but handle it
                    patched_activation[:, patch_pos] = biased_activation[:, biased_patch_pos]
                elif len(activation.shape) == 3:  # [batch, seq, features]
                    patched_activation[:, patch_pos, :] = biased_activation[:, biased_patch_pos, :]
                elif len(activation.shape) == 4:  # [batch, seq, heads, head_dim]
                    patched_activation[:, patch_pos, :, :] = biased_activation[:, biased_patch_pos, :, :]
                else:
                    # For other shapes, try to patch appropriately
                    # Use indexing that works for most cases
                    idx = (slice(None), patch_pos) + (slice(None),) * (len(activation.shape) - 2)
                    biased_idx = (slice(None), biased_patch_pos) + (slice(None),) * (len(biased_activation.shape) - 2)
                    patched_activation[idx] = biased_activation[biased_idx]
            
            return patched_activation
        return activation
    
    hooks = [(hook_name, patch_hook)]
    
    # Run context with patched activation
    with torch.no_grad():
        patched_logits = model.run_with_hooks(context_tokens, fwd_hooks=hooks)
        patched_bias = bias_metric_fn(patched_logits)
    
    # Return absolute change in bias metric
    return abs(patched_bias - original_bias)


def causal_patch_head(
    model: HookedTransformer,
    biased_tokens: torch.Tensor,
    neutral_tokens: torch.Tensor,
    layer: int,
    head: int,
    bias_metric_fn: Callable[[torch.Tensor], float]
) -> float:
    """
    Perform causal patching on a specific attention head.
    
    Args:
        model: HookedTransformer model
        biased_tokens: Tokens from biased prompt
        neutral_tokens: Tokens from neutral prompt
        layer: Layer index
        head: Head index within layer
        bias_metric_fn: Function that computes bias from logits
    
    Returns:
        Absolute change in bias metric
    """
    hook_name = f"blocks.{layer}.attn.hook_z"
    
    # Get biased activations
    with torch.no_grad():
        _, biased_cache = model.run_with_cache(biased_tokens)
    
    # Get original bias
    with torch.no_grad():
        original_logits = model(neutral_tokens)
        original_bias = bias_metric_fn(original_logits)
    
    # Patch specific head
    def patch_head_hook(activation: torch.Tensor, hook: HookPoint):
        # activation shape: [batch, seq, n_heads, head_dim]
        if hook_name in biased_cache:
            biased_activation = biased_cache[hook_name]
            # Copy only the specific head
            activation = activation.clone()
            activation[:, :, head, :] = biased_activation[:, :, head, :]
        return activation
    
    hooks = [(hook_name, patch_head_hook)]
    
    with torch.no_grad():
        patched_logits = model.run_with_hooks(neutral_tokens, fwd_hooks=hooks)
        patched_bias = bias_metric_fn(patched_logits)
    
    return abs(patched_bias - original_bias)


def scan_all_edges(
    model: HookedTransformer,
    biased_examples: List[torch.Tensor],
    neutral_examples: List[torch.Tensor],
    bias_metric_fn: Callable[[torch.Tensor], float],
    hook_points: List[str] = None,
    context_len: int = None
) -> Dict[str, float]:
    """
    Scan all edges and compute their impact on bias metric.
    
    Following Chandna et al. (2025): systematically patch each edge
    and measure the change in bias metric.
    
    Args:
        model: HookedTransformer model
        biased_examples: List of tokenized biased prompts
        neutral_examples: List of tokenized neutral prompts
        bias_metric_fn: Function that computes bias from logits
        hook_points: Optional list of hook points to scan (default: all)
        context_len: Length of context (for proper patching position)
    
    Returns:
        Dictionary mapping hook point names to impact scores
    """
    if hook_points is None:
        hook_points = get_all_hook_points(model)
    
    # Use first example pair for scanning (can be extended to average over multiple)
    biased_tokens = biased_examples[0]
    neutral_tokens = neutral_examples[0]
    
    # Ensure they're tensors
    if not isinstance(biased_tokens, torch.Tensor):
        biased_tokens = torch.tensor(biased_tokens)
    if not isinstance(neutral_tokens, torch.Tensor):
        neutral_tokens = torch.tensor(neutral_tokens)
    
    biased_tokens = biased_tokens.to(model.cfg.device)
    neutral_tokens = neutral_tokens.to(model.cfg.device)
    
    impact_scores = {}
    
    for hook_name in hook_points:
        try:
            impact = causal_patch_edge(
                model, biased_tokens, neutral_tokens, hook_name, bias_metric_fn, context_len
            )
            impact_scores[hook_name] = impact
        except Exception as e:
            # If patching fails for this hook, skip it
            impact_scores[hook_name] = 0.0
    
    return impact_scores


def scan_all_heads(
    model: HookedTransformer,
    biased_examples: List[torch.Tensor],
    neutral_examples: List[torch.Tensor],
    bias_metric_fn: Callable[[torch.Tensor], float]
) -> Dict[Tuple[int, int], float]:
    """
    Scan all attention heads and compute their impact on bias.
    
    Args:
        model: HookedTransformer model
        biased_examples: List of tokenized biased prompts
        neutral_examples: List of tokenized neutral prompts
        bias_metric_fn: Function that computes bias from logits
    
    Returns:
        Dictionary mapping (layer, head) tuples to impact scores
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    
    biased_tokens = biased_examples[0].unsqueeze(0) if biased_examples[0].dim() == 1 else biased_examples[0]
    neutral_tokens = neutral_examples[0].unsqueeze(0) if neutral_examples[0].dim() == 1 else neutral_examples[0]
    
    biased_tokens = biased_tokens.to(model.cfg.device)
    neutral_tokens = neutral_tokens.to(model.cfg.device)
    
    impact_scores = {}
    
    for layer in range(n_layers):
        for head in range(n_heads):
            try:
                impact = causal_patch_head(
                    model, biased_tokens, neutral_tokens, layer, head, bias_metric_fn
                )
                impact_scores[(layer, head)] = impact
            except Exception:
                impact_scores[(layer, head)] = 0.0
    
    return impact_scores
