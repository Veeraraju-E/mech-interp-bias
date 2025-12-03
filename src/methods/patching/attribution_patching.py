"""Attribution patching: gradient-based edge importance estimation (Nanda 2023)."""

import torch
from typing import Dict, List, Callable, Any
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint
from tqdm import tqdm
from .hook_points import get_all_hook_points

def save_activation(name: str, activations: Dict[str, torch.Tensor]):
    def hook_fn(activation: torch.Tensor, hook: HookPoint):
        activation.retain_grad()
        activations[name] = activation
        return activation
    return hook_fn

def attribution_patch(model: HookedTransformer, examples: List[Dict[str, Any]], bias_metric_fn: Callable[[torch.Tensor, Dict[str, Any]], torch.Tensor], hook_points: List[str] = None) -> Dict[str, float]:
    """
    Compute attribution scores: grad(bias_metric) @ activation for each edge.
    
    Attribution is computed at the logit_position (specified in metadata).
    """
    if hook_points is None:
        hook_points = get_all_hook_points(model)
    
    all_attributions = {hook_name: [] for hook_name in hook_points}
    model.eval()
    
    for example in tqdm(examples, desc="Computing attributions"):
        tokens = example["tokens"]
        metadata = example.get("metadata", {})
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
        tokens = tokens.to(model.cfg.device)
        tokens.requires_grad_(False)
        
        logit_position = metadata.get("logit_position")
        if logit_position is None:
            if tokens.dim() == 2:
                logit_position = tokens.shape[1] - 1
            else:
                logit_position = tokens.shape[0] - 1
            # print(f"Using last logit position since unspecified: {logit_position}")
        
        activations = {}
        hooks = [(hook_name, save_activation(hook_name, activations)) for hook_name in hook_points]
        
        with torch.enable_grad():
            logits = model.run_with_hooks(tokens, fwd_hooks=hooks)
            bias_score = bias_metric_fn(logits, metadata)
        
        if isinstance(bias_score, torch.Tensor):
            if bias_score.numel() > 1:
                bias_score = bias_score.mean()
            if bias_score.grad_fn is None:
                continue
        else:
            bias_score = torch.tensor(float(bias_score), device=logits.device, requires_grad=True)
            if bias_score.grad_fn is None:
                continue
        
        if bias_score.requires_grad and bias_score.grad_fn:
            bias_score.backward(retain_graph=False)
        else:
            continue
        
        for hook_name in hook_points:
            if hook_name in activations:
                activation = activations[hook_name]
                if activation.grad is not None:
                    if activation.dim() == 3:
                        seq_len = activation.shape[1]
                        pos = max(0, min(seq_len - 1, logit_position))
                        grad_at_pos = activation.grad[:, pos, :]
                        act_at_pos = activation.detach()[:, pos, :]
                        attribution = (grad_at_pos * act_at_pos).sum().item()
                    else:
                        attribution = (activation.grad * activation.detach()).sum().item()
                    all_attributions[hook_name].append(abs(attribution))
                else:
                    all_attributions[hook_name].append(0.0)
            else:
                all_attributions[hook_name].append(0.0)
        
        for activation in activations.values():
            if activation.grad is not None:
                activation.grad.zero_()
    
    attributions = {}
    for hook_name, scores in all_attributions.items(): attributions[hook_name] = sum(scores) / len(scores) if scores else 0.0
    
    return attributions