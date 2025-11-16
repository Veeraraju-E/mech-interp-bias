"""Hook point utilities for attribution patching and ablations."""

from typing import List
from transformer_lens import HookedTransformer

def get_all_hook_points(model: HookedTransformer) -> List[str]:
    """Get all hook points for attention heads, MLPs, and residual streams."""
    hook_points = []
    n_layers = model.cfg.n_layers
    
    for layer in range(n_layers):
        hook_points.append(f"blocks.{layer}.attn.hook_z")
        hook_points.append(f"blocks.{layer}.hook_mlp_out")
        hook_points.append(f"blocks.{layer}.hook_resid_post")
    
    return hook_points
