from typing import List, Dict, Any, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from .sae_model import SparseAutoencoder


def compute_latent_correlations(sae: SparseAutoencoder, activations: np.ndarray, labels: np.ndarray, batch_size: int = 512) -> List[Dict[str, Any]]:
    """Compute Pearson correlations between SAE latents and GLD labels."""
    device = sae.device
    dataset = TensorDataset(torch.from_numpy(activations).float())
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    latents_list: List[np.ndarray] = []
    with torch.no_grad():
        for batch, in tqdm(dataloader, desc="Encoding activations"):
            encoded = sae.encode(batch.to(device), apply_sparsity=True)
            latents_list.append(encoded.cpu().numpy())

    latents = np.concatenate(latents_list, axis=0)
    correlations: List[Dict[str, Any]] = []
    labels_centered = labels - labels.mean()
    label_std = labels_centered.std() + 1e-8

    for idx in range(latents.shape[1]):
        latent = latents[:, idx]
        latent_centered = latent - latent.mean()
        latent_std = latent_centered.std() + 1e-8
        corr = float((latent_centered * labels_centered).mean() / (latent_std * label_std))
        correlations.append(
            {
                "latent": idx,
                "correlation": corr,
                "mean_activation": float(latent.mean()),
            }
        )
    correlations.sort(key=lambda x: abs(x["correlation"]), reverse=True)
    return correlations


def select_bias_latents(correlations: List[Dict[str, Any]], min_abs_corr: float = 0.05, top_k: Optional[int] = None) -> List[int]:
    """Return latent indices exceeding correlation threshold."""
    filtered = [c for c in correlations if abs(c["correlation"]) >= min_abs_corr]
    if top_k:
        filtered = filtered[:top_k]
    return [c["latent"] for c in filtered]

