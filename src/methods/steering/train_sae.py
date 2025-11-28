from typing import Dict, Any, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from .sae_model import SparseAutoencoder


def train_sae(
    activations: np.ndarray,
    d_model: int,
    d_latent: int,
    k_sparse: int,
    device: torch.device,
    batch_size: int = 1024,
    epochs: int = 10,
    lr: float = 1e-4,
    lambda_l1: float = 0.0,
) -> Tuple[SparseAutoencoder, Dict[str, Any]]:
    """Train a Sparse Autoencoder on pre-collected activations."""
    sae = SparseAutoencoder(d_model=d_model, d_latent=d_latent or (d_model * 8), k_sparse=k_sparse, device=device)
    dataset = TensorDataset(torch.from_numpy(activations).float())
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(sae.parameters(), lr=lr)

    activation_counts = torch.zeros(sae.d_latent, device=device)

    total_steps = epochs * len(dataloader)
    progress = tqdm(total=total_steps, desc="SAE training", dynamic_ncols=True, leave=False)

    for epoch in range(epochs):
        for batch, in dataloader:
            batch = batch.to(device)
            loss, recon, sparsity = sae.compute_loss(batch, lambda_l1=lambda_l1)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(sae.parameters(), max_norm=1.0)
            optimizer.step()
            sae._normalize_decoder()

            with torch.no_grad():
                latents = sae.encode(batch, apply_sparsity=True)
                activation_counts += (latents > 1e-8).sum(dim=0).float()

            dead_latents = int((activation_counts == 0).sum().item())
            progress.set_postfix(
                {
                    "epoch": f"{epoch+1}/{epochs}",
                    "loss": f"{loss.item():.4f}",
                    "dead": f"{dead_latents}/{sae.d_latent}",
                }
            )
            progress.update(1)

    progress.close()

    return sae, {}