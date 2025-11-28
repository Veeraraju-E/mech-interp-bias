"""Analyze which samples maximally activate top SAE latents for biased prompts."""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import torch
from tqdm import tqdm

from src.model_setup import load_model, get_tokenizer, setup_device
from .dataset import collect_activation_dataset, build_gender_prompts, compute_gld_scores
from .sae_model import SparseAutoencoder
from .train_sae import train_sae


def find_top_biased_latents(sae: SparseAutoencoder, biased_activations: np.ndarray, top_k: int = 10) -> List[Tuple[int, float]]:
    device = sae.device
    biased_tensor = torch.from_numpy(biased_activations).float().to(device)
    
    with torch.no_grad():
        latents = sae.encode(biased_tensor, apply_sparsity=True)  # [n_biased, d_latent]
    
    mean_activations = latents.mean(dim=0).cpu().numpy()  # [d_latent]
    
    top_indices = np.argsort(mean_activations)[::-1][:top_k]
    top_latents = [(int(idx), float(mean_activations[idx])) for idx in top_indices]
    
    return top_latents


def find_top_samples_for_latent(sae: SparseAutoencoder, activations: np.ndarray, sample_metadata: List[Dict[str, Any]], latent_idx: int, top_k: int = 20) -> List[Dict[str, Any]]:
    device = sae.device
    activations_tensor = torch.from_numpy(activations).float().to(device)
    
    with torch.no_grad():
        latents = sae.encode(activations_tensor, apply_sparsity=True)  # [n_samples, d_latent]
    
    latent_activations = latents[:, latent_idx].cpu().numpy()  # [n_samples]
    
    top_indices = np.argsort(latent_activations)[::-1][:top_k]
    
    top_samples = []
    for idx in top_indices:
        sample_info = sample_metadata[idx].copy()
        sample_info["activation_value"] = float(latent_activations[idx])
        sample_info["sample_index"] = int(idx)
        top_samples.append(sample_info)
    
    return top_samples


def analyze_top_latent_samples(
    model_name: str,
    dataset_name: str,
    layer: int,
    sae_path: Optional[Path] = None,
    activations: Optional[np.ndarray] = None,
    sample_metadata: Optional[List[Dict[str, Any]]] = None,
    d_latent: Optional[int] = None,
    k_sparse: int = 64,
    top_latents: int = 10,
    top_samples_per_latent: int = 20,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:

    device = setup_device()
    model = load_model(model_name)
    model.to(device)
    tokenizer = get_tokenizer(model_name)
    
    if output_dir is None:
        output_dir = Path("results") / model_name / "sae_bias_suppression" / "latent_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load or compute activations
    if activations is None or sample_metadata is None:
        print("Collecting activations...")
        activations, gld_scores, scored_entries = collect_activation_dataset(model=model, tokenizer=tokenizer, dataset_name=dataset_name, layer=layer)
        sample_metadata = scored_entries
    else:
        prompts = build_gender_prompts(dataset_name)
        scored_entries = compute_gld_scores(model, tokenizer, prompts)
        sample_metadata = scored_entries
    
    # Identify biased prompts (male_prob > female_prob)
    bias_mask = np.array([entry.get("male_prob", 0.0) > entry.get("female_prob", 0.0) for entry in sample_metadata])
    biased_activations = activations[bias_mask]
    biased_metadata = [sample_metadata[i] for i in range(len(sample_metadata)) if bias_mask[i]]
    
    print(f"Found {len(biased_activations)} biased prompts out of {len(activations)} total")
    
    # Load or train SAE
    if sae_path and sae_path.exists():
        print(f"Loading SAE from {sae_path}...")
        sae_data = torch.load(sae_path, map_location=device)
        if isinstance(sae_data, dict) and "state_dict" in sae_data:
            sae = SparseAutoencoder(d_model=model.cfg.d_model, d_latent=d_latent or model.cfg.d_model * 8, k_sparse=k_sparse, device=device)
            sae.load_state_dict(sae_data["state_dict"])
        elif isinstance(sae_data, SparseAutoencoder):
            sae = sae_data.to(device)
        sae.eval()
    else:
        print("Training SAE...")
        sae, _ = train_sae(activations=activations, d_model=model.cfg.d_model, d_latent=d_latent or model.cfg.d_model * 8, k_sparse=k_sparse, device=device, epochs=1000)
        sae.eval()
    
    # Find top latents for biased prompts
    print(f"Finding top {top_latents} latents for biased prompts...")
    top_latent_list = find_top_biased_latents(sae=sae, biased_activations=biased_activations, top_k=top_latents)
    
    print(f"\nTop {top_latents} latents (by mean activation on biased prompts):")
    for latent_idx, mean_act in top_latent_list:
        print(f"  Latent {latent_idx}: mean activation = {mean_act:.6f}")
    
    # For each top latent, find top samples
    results = {
        "model": model_name,
        "dataset": dataset_name,
        "layer": layer,
        "k_sparse": k_sparse,
        "d_latent": d_latent or model.cfg.d_model * 8,
        "top_latents": [],
    }
    
    print(f"\nFinding top {top_samples_per_latent} samples for each latent...")
    for latent_idx, mean_activation in tqdm(top_latent_list, desc="Analyzing latents"):
        top_samples = find_top_samples_for_latent(sae=sae, activations=activations, sample_metadata=sample_metadata, latent_idx=latent_idx, top_k=top_samples_per_latent)
        
        results["top_latents"].append({
            "latent_index": latent_idx,
            "mean_activation_on_biased": mean_activation,
            "top_samples": top_samples,
        })
    
    # Save results
    output_file = output_dir / f"top_latent_samples_layer{layer}_k{k_sparse}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nAnalysis saved to {output_file}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Analyze which samples maximally activate top SAE latents")
    parser.add_argument("--model", type=str, default="gpt2-medium")
    parser.add_argument("--dataset", type=str, default="stereoset_gender", choices=["stereoset_gender", "winogender"])
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--sae-path", type=str, default=None, help="Path to saved SAE (optional)")
    parser.add_argument("--d-latent", type=int, default=None)
    parser.add_argument("--k-sparse", type=int, default=64)
    parser.add_argument("--top-latents", type=int, default=10, help="Number of top latents to analyze")
    parser.add_argument("--top-samples", type=int, default=20, help="Number of top samples per latent")
    parser.add_argument("--output-dir", type=str, default=None)
    
    args = parser.parse_args()
    
    sae_path = Path(args.sae_path) if args.sae_path else None
    
    analyze_top_latent_samples(
        model_name=args.model,
        dataset_name=args.dataset,
        layer=args.layer,
        sae_path=sae_path,
        d_latent=args.d_latent,
        k_sparse=args.k_sparse,
        top_latents=args.top_latents,
        top_samples_per_latent=args.top_samples,
        output_dir=Path(args.output_dir) if args.output_dir else None,
    )


if __name__ == "__main__":
    main()