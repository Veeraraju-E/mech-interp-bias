from typing import List

import torch
from transformer_lens import HookedTransformer

from .sae_model import SparseAutoencoder


def create_suppression_hook(sae: SparseAutoencoder, suppressed_latents: List[int], scale: float = 0.0) -> callable:
    """Return hook that suppresses specified latent indices."""
    device = sae.device
    suppressed = torch.tensor(suppressed_latents, device=device)

    def hook_fn(activation: torch.Tensor, hook) -> torch.Tensor:
        latents = sae.encode(activation, apply_sparsity=True)
        latents[..., suppressed] *= scale
        decoded = sae.decode(latents)
        # Preserve activation norm to avoid perplexity explosions
        orig_norm = activation.norm(dim=-1, keepdim=True) + 1e-8
        dec_norm = decoded.norm(dim=-1, keepdim=True) + 1e-8
        decoded = decoded * (orig_norm / dec_norm)
        return decoded

    return hook_fn


def apply_suppression(model: HookedTransformer, tokens: torch.Tensor, sae: SparseAutoencoder, suppressed_latents: List[int], layer: int, scale: float = 0.0) -> torch.Tensor:
    """Run model with suppression hook applied at a given layer."""
    if layer == 0:
        hook_name = "blocks.0.hook_resid_pre"
    else:
        hook_name = f"blocks.{layer-1}.hook_resid_post"
    hook_fn = create_suppression_hook(sae, suppressed_latents, scale=scale)
    with torch.no_grad():
        logits = model.run_with_hooks(tokens, fwd_hooks=[(hook_name, hook_fn)])
    return logits