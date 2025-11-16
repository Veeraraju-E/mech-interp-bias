"""Attribution patching for efficient edge importance estimation.

Following Nanda (2023): Attribution Patching methodology.
Approximates edge importance via gradients instead of full activation patching.
"""

import torch
from typing import Dict, List, Callable, Tuple
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint


def attribution_patch(
    model: HookedTransformer,
    examples: List[torch.Tensor],
    bias_metric_fn: Callable[[torch.Tensor], float],
    hook_points: List[str] = None
) -> Dict[str, float]:
    """
    Compute attribution scores for edges using gradient-based approximation.
    
    Following Nanda (2023): approximates edge importance via gradients.
    Requires 1 forward pass + 1 backward pass instead of N forward passes.
    
    Attribution = grad(bias_metric) @ activation for each edge.
    
    Args:
        model: HookedTransformer model
        examples: List of tokenized input prompts
        bias_metric_fn: Function that computes bias from logits
        hook_points: Optional list of hook points to score (default: all)
    
    Returns:
        Dictionary mapping hook point names to attribution scores
    """
    from .activation_patching import get_all_hook_points
    
    if hook_points is None:
        hook_points = get_all_hook_points(model)
    
    # Use first example for attribution (can be extended to average)
    tokens = examples[0].unsqueeze(0) if examples[0].dim() == 1 else examples[0]
    tokens = tokens.to(model.cfg.device)
    tokens.requires_grad_(False)  # We don't need gradients w.r.t. input tokens
    
    # Forward pass with hooks to capture activations (with gradients enabled)
    activations = {}
    
    def save_activation(name: str):
        def hook_fn(activation: torch.Tensor, hook: HookPoint):
            # Store activation and enable gradients
            # Use retain_grad to ensure gradients are computed
            activation = activation.clone()
            activation.retain_grad()
            activations[name] = activation
            return activation
        return hook_fn
    
    hooks = []
    for hook_name in hook_points:
        hook_fn = save_activation(hook_name)
        hooks.append((hook_name, hook_fn))
    
    # Forward pass
    logits = model.run_with_hooks(tokens, fwd_hooks=hooks)
    
    # Compute bias metric
    bias_score = bias_metric_fn(logits)
    
    # Backward pass to get gradients
    if bias_score.requires_grad:
        bias_score.backward()
    else:
        # If bias_score doesn't require grad, we need to make it differentiable
        # This shouldn't happen if bias_metric_fn returns a tensor with grad
        return {hook_name: 0.0 for hook_name in hook_points}
    
    # Compute attributions: grad @ activation for each edge
    attributions = {}
    for hook_name in hook_points:
        if hook_name in activations:
            activation = activations[hook_name]
            # Get gradient w.r.t. activation
            if activation.grad is not None:
                # Attribution = sum(grad * activation) as approximation
                # Following Nanda (2023): dot product of gradient and activation
                attribution = (activation.grad * activation.detach()).sum().item()
            else:
                attribution = 0.0
            attributions[hook_name] = abs(attribution)
        else:
            attributions[hook_name] = 0.0
    
    return attributions


def validate_top_edges(
    model: HookedTransformer,
    top_edges: List[Tuple[str, float]],
    biased_examples: List[torch.Tensor],
    neutral_examples: List[torch.Tensor],
    bias_metric_fn: Callable[[torch.Tensor], float],
    context_len: int = None
) -> Dict[str, float]:
    """
    Validate top attribution edges using causal patching.
    
    Following Nanda (2023): validate gradient-based attributions with
    actual causal patching to check approximation accuracy.
    
    Args:
        model: HookedTransformer model
        top_edges: List of (hook_name, attribution_score) tuples
        biased_examples: List of tokenized biased prompts
        neutral_examples: List of tokenized neutral prompts
        bias_metric_fn: Function that computes bias from logits
    
    Returns:
        Dictionary mapping hook names to validated causal impact scores
    """
    from .activation_patching import causal_patch_edge
    
    validated_scores = {}
    
    biased_tokens = biased_examples[0].unsqueeze(0) if biased_examples[0].dim() == 1 else biased_examples[0]
    neutral_tokens = neutral_examples[0].unsqueeze(0) if neutral_examples[0].dim() == 1 else neutral_examples[0]
    
    biased_tokens = biased_tokens.to(model.cfg.device)
    neutral_tokens = neutral_tokens.to(model.cfg.device)
    
    for hook_name, _ in top_edges:
        try:
            impact = causal_patch_edge(
                model, biased_tokens, neutral_tokens, hook_name, bias_metric_fn, context_len
            )
            validated_scores[hook_name] = impact
        except Exception:
            validated_scores[hook_name] = 0.0
    
    return validated_scores
