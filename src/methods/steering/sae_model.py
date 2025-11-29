from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseAutoencoder(nn.Module):
    """k-sparse autoencoder operating on transformer residual activations."""

    def __init__(self, d_model: int, d_latent: int, k_sparse: int, device: torch.device) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_latent = d_latent
        self.k_sparse = k_sparse
        self.device = device

        self.encoder = nn.Linear(d_model, d_latent, bias=False)
        self.decoder = nn.Linear(d_latent, d_model, bias=False)
        nn.init.xavier_uniform_(self.encoder.weight)
        with torch.no_grad():
            self.decoder.weight.copy_(self.encoder.weight.t())
        self._normalize_decoder()
        self.to(device)

    def _normalize_decoder(self) -> None:
        with torch.no_grad():
            norms = torch.norm(self.decoder.weight.data, dim=0, keepdim=True) + 1e-8
            self.decoder.weight.data /= norms

    def _topk(self, latents: torch.Tensor) -> torch.Tensor:
        if self.k_sparse >= latents.shape[-1]:
            return F.relu(latents)
        values, indices = torch.topk(latents, k=self.k_sparse, dim=-1)
        sparse = torch.zeros_like(latents)
        sparse.scatter_(-1, indices, values)
        return F.relu(sparse)

    def encode(self, activations: torch.Tensor, apply_sparsity: bool = True) -> torch.Tensor:
        latents = self.encoder(activations)
        if apply_sparsity:
            latents = self._topk(latents)
        return latents

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder(latents)

    def forward(self, activations: torch.Tensor, apply_sparsity: bool = True, return_latents: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        latents = self.encode(activations, apply_sparsity=apply_sparsity)
        reconstruction = self.decode(latents)
        if return_latents:
            return reconstruction, latents
        return reconstruction

    def compute_loss(self, activations: torch.Tensor) -> Tuple[torch.Tensor, float]:
        recon_loss = F.mse_loss(self.forward(activations, apply_sparsity=True), activations)
        return recon_loss, recon_loss.item()