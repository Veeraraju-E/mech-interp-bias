from typing import List, Dict, Any, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from .sae_model import SparseAutoencoder


def compute_latent_correlations(sae: SparseAutoencoder, activations: np.ndarray, labels: np.ndarray, batch_size: int = 512) -> List[Dict[str, Any]]:
    """Compute Pearson correlations between SAE latents and GLD labels."""
    device = sae.device
    latents_list = []
    with torch.no_grad():
        for batch, in tqdm(DataLoader(TensorDataset(torch.from_numpy(activations).float()), batch_size=batch_size, shuffle=False), desc="Encoding activations"):
            latents_list.append(sae.encode(batch.to(device), apply_sparsity=True).cpu().numpy())

    latents = np.concatenate(latents_list, axis=0)
    labels_centered = labels - labels.mean()
    label_std = labels_centered.std() + 1e-8
    latents_centered = latents - latents.mean(axis=0, keepdims=True)
    latents_std = latents_centered.std(axis=0, keepdims=True) + 1e-8
    
    correlations = [
        {"latent": idx, "correlation": float((latents_centered[:, idx] * labels_centered).mean() / (latents_std[0, idx] * label_std)),
         "mean_activation": float(latents[:, idx].mean())} for idx in range(latents.shape[1])
    ]
    correlations.sort(key=lambda x: abs(x["correlation"]), reverse=True)
    return correlations


def select_bias_latents(correlations: List[Dict[str, Any]], min_abs_corr: float = 0.05, top_k: Optional[int] = None) -> List[int]:
    """Return latent indices exceeding correlation threshold."""
    filtered = [c for c in correlations if abs(c["correlation"]) >= min_abs_corr]
    if top_k:
        filtered = filtered[:top_k]
    return [c["latent"] for c in filtered]

