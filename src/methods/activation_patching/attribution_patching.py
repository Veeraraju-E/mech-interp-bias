"""Attribution patching: gradient-based edge importance estimation (Nanda 2023)."""

import torch
from typing import Dict, List, Callable
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint
from tqdm import tqdm
from .hook_points import get_all_hook_points


def attribution_patch(
    model: HookedTransformer,
    examples: List[torch.Tensor],
    bias_metric_fn: Callable[[torch.Tensor], torch.Tensor],
    hook_points: List[str] = None
) -> Dict[str, float]:
    """Compute attribution scores: grad(bias_metric) @ activation for each edge."""
    if hook_points is None:
        hook_points = get_all_hook_points(model)
    
    all_attributions = {hook_name: [] for hook_name in hook_points}
    model.eval()
    
    for tokens in tqdm(examples, desc="Computing attributions"):
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
        tokens = tokens.to(model.cfg.device)
        tokens.requires_grad_(False)
        
        activations = {}
        
        def save_activation(name: str):
            def hook_fn(activation: torch.Tensor, hook: HookPoint):
                activation.retain_grad()
                activations[name] = activation
                return activation
            return hook_fn
        
        hooks = [(hook_name, save_activation(hook_name)) for hook_name in hook_points]
        
        with torch.enable_grad():
            logits = model.run_with_hooks(tokens, fwd_hooks=hooks)
            bias_score = bias_metric_fn(logits)
        
        if isinstance(bias_score, torch.Tensor):
            if bias_score.numel() > 1:
                bias_score = bias_score.mean()
            if bias_score.grad_fn is None:
                continue
        else:
            bias_score = torch.tensor(float(bias_score), device=logits.device, requires_grad=True)
            if bias_score.grad_fn is None:
                continue
        
        if bias_score.requires_grad and bias_score.grad_fn is not None:
            bias_score.backward(retain_graph=False)
        else:
            continue
        
        for hook_name in hook_points:
            if hook_name in activations:
                activation = activations[hook_name]
                if activation.grad is not None:
                    attribution = (activation.grad * activation.detach()).sum().item()
                    all_attributions[hook_name].append(abs(attribution))
                else:
                    all_attributions[hook_name].append(0.0)
            else:
                all_attributions[hook_name].append(0.0)
        
        for activation in activations.values():
            if activation.grad is not None:
                activation.grad.zero_()
    
    attributions = {
        hook_name: sum(scores) / len(scores) if scores else 0.0
        for hook_name, scores in all_attributions.items()
    }
    
    return attributions
