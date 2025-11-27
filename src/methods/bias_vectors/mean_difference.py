"""Mean difference vectors for bias identification.

Following methodologies from:
- Gupta et al. (2025): Activation steering vectors for bias mitigation
- Jørgensen (2023): Enhancing vectors by centering
- Nanda et al. (2024): Extracting linear directions in activation space

The mean difference method computes a "bias vector" as:
    v_bias = μ(biased activations) - μ(neutral activations)

This vector represents the direction in activation space that pushes
outputs towards biased completions.
"""

import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from tqdm import tqdm
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import json

from transformer_lens import HookedTransformer
from transformers import GPT2Tokenizer


def collect_activations(
    model: HookedTransformer,
    prompts: List[str],
    tokenizer: GPT2Tokenizer,
    layer_idx: int,
    hook_name: str = "hook_resid_post",
    position: int = -1
) -> torch.Tensor:
    """
    Collect activations from a specific layer for a batch of prompts.
    
    Args:
        model: HookedTransformer model
        prompts: List of text prompts
        tokenizer: Tokenizer instance
        layer_idx: Layer index to extract activations from
        hook_name: Name of hook point (default: residual stream)
        position: Token position to extract (-1 for last token)
    
    Returns:
        Tensor of activations, shape (n_prompts, hidden_dim)
    """
    device = model.cfg.device
    activations = []
    
    for prompt in prompts:
        tokens = tokenizer.encode(prompt, return_tensors="pt").to(device)
        
        # Storage for activation
        cache = {}
        
        def hook_fn(activation, hook):
            cache['activation'] = activation.detach()
            return activation
        
        # Register hook at specific layer
        hook_point = f"blocks.{layer_idx}.{hook_name}"
        
        with torch.no_grad():
            model.run_with_hooks(
                tokens,
                fwd_hooks=[(hook_point, hook_fn)]
            )
        
        # Extract activation at specified position
        if 'activation' in cache:
            act = cache['activation']
            if act.dim() == 3:  # [batch, seq, hidden]
                act = act[0, position, :]
            elif act.dim() == 2:  # [seq, hidden]
                act = act[position, :]
            activations.append(act.cpu())
    
    return torch.stack(activations)


def compute_mean_difference_vector(
    model: HookedTransformer,
    biased_prompts: List[str],
    neutral_prompts: List[str],
    tokenizer: GPT2Tokenizer,
    layer_idx: int,
    hook_name: str = "hook_resid_post",
    position: int = -1
) -> torch.Tensor:
    """
    Compute bias vector as mean difference between biased and neutral activations.
    
    Following Gupta et al. (2025):
        v_bias = μ(biased) - μ(neutral)
    
    Args:
        model: HookedTransformer model
        biased_prompts: List of biased text prompts
        neutral_prompts: List of neutral text prompts
        tokenizer: Tokenizer instance
        layer_idx: Layer to extract vector from
        hook_name: Hook point name
        position: Token position
    
    Returns:
        Bias vector, shape (hidden_dim,)
    """
    print(f"Computing mean difference vector at layer {layer_idx}...")
    
    # Collect biased activations
    print("  Collecting biased activations...")
    biased_acts = collect_activations(
        model, biased_prompts, tokenizer, layer_idx, hook_name, position
    )
    
    # Collect neutral activations
    print("  Collecting neutral activations...")
    neutral_acts = collect_activations(
        model, neutral_prompts, tokenizer, layer_idx, hook_name, position
    )
    
    # Compute means
    biased_mean = biased_acts.mean(dim=0)
    neutral_mean = neutral_acts.mean(dim=0)
    
    # Difference vector
    bias_vector = biased_mean - neutral_mean
    
    print(f"  Bias vector norm: {bias_vector.norm().item():.4f}")
    
    return bias_vector


def enhance_bias_vector(
    bias_vector: torch.Tensor,
    biased_activations: torch.Tensor,
    neutral_activations: torch.Tensor,
    background_mean: Optional[torch.Tensor] = None,
    use_pca: bool = True
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Enhance bias vector using centering and PCA validation.
    
    Following Jørgensen (2023):
    1. Center activations by subtracting background mean
    2. Recompute difference vector
    3. Optionally validate with PCA (check if top PC aligns with vector)
    
    Args:
        bias_vector: Initial bias vector
        biased_activations: Biased activation samples
        neutral_activations: Neutral activation samples
        background_mean: Optional background mean to subtract
        use_pca: Whether to perform PCA validation
    
    Returns:
        Enhanced bias vector and analysis dictionary
    """
    analysis = {
        "original_norm": bias_vector.norm().item()
    }
    
    # Center activations if background mean provided
    if background_mean is not None:
        biased_centered = biased_activations - background_mean
        neutral_centered = neutral_activations - background_mean
        
        # Recompute difference
        enhanced_vector = biased_centered.mean(dim=0) - neutral_centered.mean(dim=0)
        analysis["centered_norm"] = enhanced_vector.norm().item()
    else:
        enhanced_vector = bias_vector
    
    # PCA validation
    if use_pca:
        # Perform PCA on biased activations
        pca = PCA(n_components=5)
        biased_np = biased_activations.numpy()
        pca.fit(biased_np)
        
        # Get top principal component
        top_pc = torch.from_numpy(pca.components_[0]).float()
        
        # Compute alignment (cosine similarity) with bias vector
        alignment = torch.nn.functional.cosine_similarity(
            enhanced_vector.unsqueeze(0),
            top_pc.unsqueeze(0)
        ).item()
        
        analysis["pca_alignment"] = alignment
        analysis["pca_explained_variance"] = pca.explained_variance_ratio_[:5].tolist()
        
        print(f"  PCA alignment: {alignment:.4f}")
        print(f"  Top PC explains {pca.explained_variance_ratio_[0]:.2%} of variance")
    
    return enhanced_vector, analysis


def extract_bias_vectors_all_layers(
    model: HookedTransformer,
    biased_prompts: List[str],
    neutral_prompts: List[str],
    tokenizer: GPT2Tokenizer,
    hook_name: str = "hook_resid_post",
    position: int = -1,
    enhance: bool = True
) -> Dict[int, Dict[str, Any]]:
    """
    Extract bias vectors from all layers of the model.
    
    Args:
        model: HookedTransformer model
        biased_prompts: List of biased prompts
        neutral_prompts: List of neutral prompts
        tokenizer: Tokenizer instance
        hook_name: Hook point name
        position: Token position
        enhance: Whether to apply enhancement (centering + PCA)
    
    Returns:
        Dictionary mapping layer index to:
            {
                'vector': bias vector,
                'norm': vector norm,
                'analysis': enhancement analysis (if enhance=True)
            }
    """
    n_layers = model.cfg.n_layers
    results = {}
    
    print(f"\n{'='*60}")
    print("Extracting Bias Vectors from All Layers")
    print(f"{'='*60}")
    print(f"Model: {model.cfg.model_name}")
    print(f"Layers: {n_layers}")
    print(f"Biased prompts: {len(biased_prompts)}")
    print(f"Neutral prompts: {len(neutral_prompts)}")
    print(f"Enhancement: {enhance}")
    print(f"{'='*60}\n")
    
    # Optionally compute background mean across all prompts
    background_mean = None
    if enhance:
        print("Computing background mean...")
        all_prompts = biased_prompts + neutral_prompts
        # Use middle layer for background
        mid_layer = n_layers // 2
        all_acts = collect_activations(
            model, all_prompts, tokenizer, mid_layer, hook_name, position
        )
        background_mean = all_acts.mean(dim=0)
        print(f"Background mean computed from {len(all_prompts)} prompts at layer {mid_layer}\n")
    
    for layer_idx in tqdm(range(n_layers), desc="Extracting vectors"):
        # Compute base vector
        bias_vector = compute_mean_difference_vector(
            model, biased_prompts, neutral_prompts, tokenizer,
            layer_idx, hook_name, position
        )
        
        layer_result = {
            'vector': bias_vector,
            'norm': bias_vector.norm().item()
        }
        
        # Enhance if requested
        if enhance:
            # Collect activations for enhancement
            biased_acts = collect_activations(
                model, biased_prompts, tokenizer, layer_idx, hook_name, position
            )
            neutral_acts = collect_activations(
                model, neutral_prompts, tokenizer, layer_idx, hook_name, position
            )
            
            enhanced_vector, analysis = enhance_bias_vector(
                bias_vector, biased_acts, neutral_acts,
                background_mean, use_pca=True
            )
            
            layer_result['enhanced_vector'] = enhanced_vector
            layer_result['enhanced_norm'] = enhanced_vector.norm().item()
            layer_result['analysis'] = analysis
        
        results[layer_idx] = layer_result
    
    print(f"\n{'='*60}")
    print("Bias vector extraction complete!")
    print(f"{'='*60}\n")
    
    return results


def visualize_bias_vectors(
    vectors_dict: Dict[int, Dict[str, Any]],
    output_dir: Path = Path("results/bias_vectors"),
    save_prefix: str = "bias_vectors"
):
    """
    Visualize bias vector properties across layers.
    
    Args:
        vectors_dict: Results from extract_bias_vectors_all_layers
        output_dir: Directory to save plots
        save_prefix: Prefix for output files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    layers = sorted(vectors_dict.keys())
    norms = [vectors_dict[l]['norm'] for l in layers]
    
    # Check if enhanced vectors exist
    has_enhanced = 'enhanced_norm' in vectors_dict[layers[0]]
    if has_enhanced:
        enhanced_norms = [vectors_dict[l]['enhanced_norm'] for l in layers]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Left plot: Vector norms
    ax1.plot(layers, norms, marker='o', linewidth=2.5, label='Original Vector', 
            color='blue', linestyle='--', alpha=0.8, markersize=7)
    
    if has_enhanced:
        ax1.plot(layers, enhanced_norms, marker='s', linewidth=2, label='Enhanced Vector', 
                color='red', alpha=0.9, markersize=6)
    
    ax1.set_xlabel("Layer", fontsize=12)
    ax1.set_ylabel("Vector Norm", fontsize=12)
    ax1.set_title("Bias Vector Norms Across Layers", fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Right plot: Relative difference (%)
    if has_enhanced:
        relative_diff = [(enh - orig) / orig * 100 for orig, enh in zip(norms, enhanced_norms)]
        ax2.plot(layers, relative_diff, marker='o', linewidth=2, color='green')
        ax2.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax2.set_xlabel("Layer", fontsize=12)
        ax2.set_ylabel("Relative Difference (%)", fontsize=12)
        ax2.set_title("Enhancement Effect (% Change in Norm)", fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / f"{save_prefix}_norms.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    # If PCA alignment available, plot it
    if has_enhanced and 'analysis' in vectors_dict[layers[0]]:
        alignments = []
        for l in layers:
            analysis = vectors_dict[l].get('analysis', {})
            alignments.append(analysis.get('pca_alignment', 0.0))
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.plot(layers, alignments, marker='o', linewidth=2, color='green')
        ax.set_xlabel("Layer", fontsize=12)
        ax.set_ylabel("PCA Alignment (Cosine Similarity)", fontsize=12)
        ax.set_title("Bias Vector Alignment with Top Principal Component", fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(output_dir / f"{save_prefix}_pca_alignment.png", dpi=300, bbox_inches="tight")
        plt.close()
    
    print(f"Bias vector visualizations saved to {output_dir}")
    
    # Save vector norms to JSON
    summary = {
        "layers": layers,
        "norms": norms,
        "max_norm_layer": int(layers[np.argmax(norms)]),
        "max_norm": float(max(norms))
    }
    
    if has_enhanced:
        summary["enhanced_norms"] = enhanced_norms
        summary["max_enhanced_norm_layer"] = int(layers[np.argmax(enhanced_norms)])
        summary["max_enhanced_norm"] = float(max(enhanced_norms))
    
    with open(output_dir / f"{save_prefix}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

