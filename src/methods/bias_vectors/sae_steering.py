"""SAE-based projection steering for bias mitigation.

Uses Sparse Autoencoders (SAE) to learn a better bias direction in activation space,
then applies projection-based steering using this learned direction.

Following methodologies from:
- Anthropic's SAE work on interpretability
- Sparse feature discovery in activation space
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from tqdm import tqdm
import json

from transformer_lens import HookedTransformer
from transformers import GPT2Tokenizer

from .projection_steering import (
    orthogonal_projection,
    projection_steering_hook,
    evaluate_projection_steering,
    batch_projection_steering_experiment,
    visualize_projection_steering_results,
    generate_projection_steering_examples
)


class SparseAutoencoder(nn.Module):
    """
    Sparse Autoencoder for learning sparse features in activation space.
    
    Architecture:
    - Encoder: Linear(d_model -> n_features) with ReLU
    - Decoder: Linear(n_features -> d_model)
    - L1 sparsity penalty on features
    """
    
    def __init__(self, d_model: int, n_features: int, sparsity_coeff: float = 0.01):
        super().__init__()
        self.d_model = d_model
        self.n_features = n_features
        self.sparsity_coeff = sparsity_coeff
        
        # Encoder: activation -> features
        self.encoder = nn.Linear(d_model, n_features, bias=False)
        
        # Decoder: features -> activation
        self.decoder = nn.Linear(n_features, d_model, bias=False)
        
        # Initialize decoder weights as transpose of encoder (for better initialization)
        with torch.no_grad():
            self.decoder.weight.data = self.encoder.weight.data.T.clone()
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input activations [batch, d_model] or [seq_len, d_model]
        
        Returns:
            reconstructed: Reconstructed activations
            features: Sparse feature activations [batch, n_features] or [seq_len, n_features]
        """
        # Encode
        features = F.relu(self.encoder(x))  # [batch, n_features]
        
        # Decode
        reconstructed = self.decoder(features)  # [batch, d_model]
        
        return reconstructed, features
    
    def compute_loss(
        self,
        x: torch.Tensor,
        reconstructed: torch.Tensor,
        features: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute reconstruction loss + sparsity penalty.
        
        Args:
            x: Original activations
            reconstructed: Reconstructed activations
            features: Feature activations
        
        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary with individual loss components
        """
        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(reconstructed, x)
        
        # Sparsity penalty (L1 on features)
        sparsity_loss = self.sparsity_coeff * torch.mean(torch.abs(features))
        
        # Total loss
        total_loss = recon_loss + sparsity_loss
        
        loss_dict = {
            "reconstruction_loss": recon_loss.item(),
            "sparsity_loss": sparsity_loss.item(),
            "total_loss": total_loss.item()
        }
        
        return total_loss, loss_dict


def train_sae_on_activations(
    activations: torch.Tensor,
    d_model: int,
    n_features: int = 512,
    sparsity_coeff: float = 0.01,
    n_epochs: int = 100,
    batch_size: int = 256,
    lr: float = 1e-3,
    device: str = "cuda"
) -> SparseAutoencoder:
    """
    Train a Sparse Autoencoder on activation data.
    
    Args:
        activations: Activation tensor [n_samples, d_model]
        d_model: Model dimension
        n_features: Number of SAE features to learn
        sparsity_coeff: L1 sparsity penalty coefficient
        n_epochs: Number of training epochs
        batch_size: Batch size for training
        lr: Learning rate
        device: Device to train on
    
    Returns:
        Trained SAE model
    """
    sae = SparseAutoencoder(d_model, n_features, sparsity_coeff).to(device)
    optimizer = torch.optim.Adam(sae.parameters(), lr=lr)
    
    # Prepare data
    n_samples = activations.shape[0]
    activations = activations.to(device)
    
    print(f"\nTraining SAE on {n_samples} activations...")
    print(f"  d_model: {d_model}, n_features: {n_features}")
    print(f"  sparsity_coeff: {sparsity_coeff}, epochs: {n_epochs}\n")
    
    # Training loop
    for epoch in tqdm(range(n_epochs), desc="Training SAE"):
        # Shuffle data
        indices = torch.randperm(n_samples, device=device)
        
        epoch_losses = []
        
        for i in range(0, n_samples, batch_size):
            batch_indices = indices[i:i+batch_size]
            batch = activations[batch_indices]
            
            # Forward pass
            reconstructed, features = sae(batch)
            
            # Compute loss
            loss, loss_dict = sae.compute_loss(batch, reconstructed, features)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_losses.append(loss_dict)
        
        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            avg_loss = np.mean([l['total_loss'] for l in epoch_losses])
            avg_recon = np.mean([l['reconstruction_loss'] for l in epoch_losses])
            avg_sparse = np.mean([l['sparsity_loss'] for l in epoch_losses])
            print(f"  Epoch {epoch+1}/{n_epochs}: Loss={avg_loss:.6f} "
                  f"(Recon={avg_recon:.6f}, Sparse={avg_sparse:.6f})")
    
    print("✅ SAE training complete!\n")
    return sae


def extract_bias_direction_from_sae(
    sae: SparseAutoencoder,
    biased_activations: torch.Tensor,
    neutral_activations: torch.Tensor,
    device: str = "cuda"
) -> torch.Tensor:
    """
    Extract bias direction using SAE features.
    
    Strategy:
    1. Encode biased and neutral activations to SAE features
    2. Compute mean difference in feature space
    3. Decode the difference to get bias direction in activation space
    
    Args:
        sae: Trained Sparse Autoencoder
        biased_activations: Activations from biased prompts [n_biased, d_model]
        neutral_activations: Activations from neutral prompts [n_neutral, d_model]
        device: Device
    
    Returns:
        Bias direction vector [d_model]
    """
    sae.eval()
    biased_activations = biased_activations.to(device)
    neutral_activations = neutral_activations.to(device)
    
    with torch.no_grad():
        # Encode to feature space
        biased_features = F.relu(sae.encoder(biased_activations))
        neutral_features = F.relu(sae.encoder(neutral_activations))
        
        # Compute mean difference in feature space
        biased_mean_features = biased_features.mean(dim=0)  # [n_features]
        neutral_mean_features = neutral_features.mean(dim=0)  # [n_features]
        feature_diff = biased_mean_features - neutral_mean_features  # [n_features]
        
        # Decode feature difference to activation space
        bias_direction = sae.decoder(feature_diff.unsqueeze(0)).squeeze(0)  # [d_model]
        
        # Normalize
        bias_norm = torch.norm(bias_direction)
        if bias_norm > 1e-10:
            bias_direction = bias_direction / bias_norm
    
    return bias_direction


def extract_sae_bias_vector(
    model: HookedTransformer,
    biased_prompts: List[str],
    neutral_prompts: List[str],
    tokenizer: GPT2Tokenizer,
    layer_idx: int,
    hook_name: str = "hook_resid_post",
    position: int = -1,
    n_features: int = 512,
    sparsity_coeff: float = 0.01,
    n_epochs: int = 100,
    batch_size: int = 256,
    lr: float = 1e-3
) -> Tuple[torch.Tensor, SparseAutoencoder]:
    """
    Extract bias vector using SAE-based method.
    
    Steps:
    1. Collect activations from biased and neutral prompts
    2. Train SAE on combined activations
    3. Extract bias direction using SAE features
    
    Args:
        model: HookedTransformer model
        biased_prompts: List of biased prompts
        neutral_prompts: List of neutral prompts
        tokenizer: Tokenizer instance
        layer_idx: Layer to extract activations from
        hook_name: Hook point name
        position: Token position to extract (-1 for last token)
        n_features: Number of SAE features
        sparsity_coeff: SAE sparsity coefficient
        n_epochs: Training epochs
        batch_size: Training batch size
        lr: Learning rate
    
    Returns:
        bias_vector: Learned bias direction [d_model]
        sae: Trained SAE model
    """
    device = model.cfg.device
    hook_point = f"blocks.{layer_idx}.{hook_name}"
    
    print(f"\n{'='*60}")
    print(f"Extracting SAE-based Bias Vector (Layer {layer_idx})")
    print(f"{'='*60}\n")
    
    # Collect activations
    biased_activations = []
    neutral_activations = []
    
    print("Collecting activations...")
    for prompt in tqdm(biased_prompts, desc="Biased prompts"):
        tokens = tokenizer.encode(prompt, return_tensors="pt").to(device)
        
        def collect_hook(activation, hook):
            if activation.dim() == 3:
                act = activation[0, position, :].detach()
            else:
                act = activation[position, :].detach()
            biased_activations.append(act.cpu())
            return activation
        
        with torch.no_grad():
            model.run_with_hooks(tokens, fwd_hooks=[(hook_point, collect_hook)])
    
    for prompt in tqdm(neutral_prompts, desc="Neutral prompts"):
        tokens = tokenizer.encode(prompt, return_tensors="pt").to(device)
        
        def collect_hook(activation, hook):
            if activation.dim() == 3:
                act = activation[0, position, :].detach()
            else:
                act = activation[position, :].detach()
            neutral_activations.append(act.cpu())
            return activation
        
        with torch.no_grad():
            model.run_with_hooks(tokens, fwd_hooks=[(hook_point, collect_hook)])
    
    # Stack activations
    biased_acts = torch.stack(biased_activations).to(device)  # [n_biased, d_model]
    neutral_acts = torch.stack(neutral_activations).to(device)  # [n_neutral, d_model]
    all_acts = torch.cat([biased_acts, neutral_acts], dim=0)  # [n_total, d_model]
    
    d_model = all_acts.shape[1]
    print(f"\nCollected {len(biased_acts)} biased and {len(neutral_acts)} neutral activations")
    print(f"Activation dimension: {d_model}")
    
    # Train SAE
    sae = train_sae_on_activations(
        all_acts,
        d_model=d_model,
        n_features=n_features,
        sparsity_coeff=sparsity_coeff,
        n_epochs=n_epochs,
        batch_size=batch_size,
        lr=lr,
        device=device
    )
    
    # Extract bias direction
    print("Extracting bias direction from SAE...")
    bias_vector = extract_bias_direction_from_sae(
        sae, biased_acts, neutral_acts, device=device
    )
    
    print(f"✅ SAE bias vector extracted (norm: {torch.norm(bias_vector).item():.4f})")
    
    return bias_vector, sae


def batch_sae_steering_experiment(
    model: HookedTransformer,
    prompts: List[Dict[str, Any]],
    tokenizer: GPT2Tokenizer,
    biased_prompts: List[str],
    neutral_prompts: List[str],
    target_layer: int,
    bias_critical_layers: Optional[List[int]] = None,
    hook_name: str = "hook_resid_post",
    position: int = -1,
    n_features: int = 512,
    sparsity_coeff: float = 0.01,
    n_epochs: int = 100,
    batch_size: int = 256,
    lr: float = 1e-3
) -> Dict[str, Any]:
    """
    Run SAE-based projection steering experiments.
    
    Args:
        model: HookedTransformer model
        prompts: List of prompt dictionaries for evaluation
        tokenizer: Tokenizer instance
        biased_prompts: Biased prompts for SAE training
        neutral_prompts: Neutral prompts for SAE training
        target_layer: Primary layer to test
        bias_critical_layers: Optional list of layers to test
        hook_name: Hook point name
        position: Token position
        n_features: SAE features
        sparsity_coeff: SAE sparsity
        n_epochs: SAE training epochs
        batch_size: SAE training batch size
        lr: SAE learning rate
    
    Returns:
        Results dictionary with SAE vectors and steering effects
    """
    # Determine layers to test
    if bias_critical_layers is None:
        layers_to_test = [target_layer]
    else:
        layers_to_test = bias_critical_layers
    
    print(f"\n{'='*60}")
    print(f"SAE-Based Projection Steering Experiment")
    print(f"{'='*60}")
    print(f"Training prompts: {len(biased_prompts)} biased, {len(neutral_prompts)} neutral")
    print(f"Evaluation prompts: {len(prompts)}")
    print(f"Layers to test: {layers_to_test}")
    print(f"{'='*60}\n")
    
    results = {}
    sae_vectors = {}
    
    for layer_idx in layers_to_test:
        print(f"\n{'='*60}")
        print(f"Processing Layer {layer_idx}")
        print(f"{'='*60}\n")
        
        # Extract SAE bias vector
        sae_vector, sae_model = extract_sae_bias_vector(
            model=model,
            biased_prompts=biased_prompts,
            neutral_prompts=neutral_prompts,
            tokenizer=tokenizer,
            layer_idx=layer_idx,
            hook_name=hook_name,
            position=position,
            n_features=n_features,
            sparsity_coeff=sparsity_coeff,
            n_epochs=n_epochs,
            batch_size=batch_size,
            lr=lr
        )
        
        sae_vectors[layer_idx] = sae_vector
        
        # Run projection steering with SAE vector
        bias_vectors_dict = {layer_idx: sae_vector}
        steering_results = batch_projection_steering_experiment(
            model=model,
            prompts=prompts,
            tokenizer=tokenizer,
            bias_vectors=bias_vectors_dict,
            target_layer=layer_idx,
            bias_critical_layers=[layer_idx],
            hook_name=hook_name,
            position=position
        )
        
        results[layer_idx] = steering_results.get(layer_idx, {})
        results[layer_idx]['sae_vector_norm'] = torch.norm(sae_vector).item()
    
    return {
        "results": results,
        "sae_vectors": {k: v.cpu() for k, v in sae_vectors.items()}
    }

