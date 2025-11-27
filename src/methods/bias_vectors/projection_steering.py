"""Projection-based bias mitigation methods.

Implements:
1. Projection-Based Steering (Section 4.1): Orthogonal projection to remove bias component
2. SAE-Based Projection Steering (Section 4.2): Using sparse autoencoders to find better bias direction

Following methodologies from:
- Gupta et al. (2025): Activation steering for bias mitigation
- ICML 2025: No Training Wheels - steering at inference time
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Callable
from tqdm import tqdm
import matplotlib.pyplot as plt
import json

from transformer_lens import HookedTransformer
from transformers import GPT2Tokenizer


def orthogonal_projection(
    activation: torch.Tensor,
    bias_vector: torch.Tensor
) -> torch.Tensor:
    """
    Apply orthogonal projection to remove bias component from activation.
    
    Formula: h̃ = h - (⟨h, v_bias⟩ / ||v_bias||²) * v_bias
    
    This projects the activation onto the orthogonal complement of the bias vector,
    effectively removing the bias-aligned component.
    
    Args:
        activation: Activation tensor [batch, seq_len, d_model] or [seq_len, d_model]
        bias_vector: Bias direction vector [d_model]
    
    Returns:
        Projected activation with bias component removed
    """
    # Ensure bias_vector is normalized direction
    bias_norm_sq = torch.dot(bias_vector, bias_vector)
    
    if bias_norm_sq < 1e-10:
        # If bias vector is too small, return original activation
        return activation
    
    # Handle different tensor shapes
    if activation.dim() == 3:
        # [batch, seq_len, d_model]
        batch_size, seq_len, d_model = activation.shape
        activation_flat = activation.view(-1, d_model)  # [batch*seq_len, d_model]
        
        # Compute dot product: ⟨h, v_bias⟩
        dot_product = torch.matmul(activation_flat, bias_vector)  # [batch*seq_len]
        
        # Compute projection: (dot_product / ||v_bias||²) * v_bias
        projection_coeff = dot_product / bias_norm_sq  # [batch*seq_len]
        projection = projection_coeff.unsqueeze(-1) * bias_vector.unsqueeze(0)  # [batch*seq_len, d_model]
        
        # Subtract projection
        projected = activation_flat - projection  # [batch*seq_len, d_model]
        return projected.view(batch_size, seq_len, d_model)
    
    elif activation.dim() == 2:
        # [seq_len, d_model]
        seq_len, d_model = activation.shape
        
        # Compute dot product
        dot_product = torch.matmul(activation, bias_vector)  # [seq_len]
        
        # Compute projection
        projection_coeff = dot_product / bias_norm_sq  # [seq_len]
        projection = projection_coeff.unsqueeze(-1) * bias_vector.unsqueeze(0)  # [seq_len, d_model]
        
        # Subtract projection
        projected = activation - projection
        return projected
    
    else:
        raise ValueError(f"Unsupported activation shape: {activation.shape}")


def projection_steering_hook(
    bias_vector: torch.Tensor,
    position: int = -1
) -> Callable:
    """
    Create a hook function for projection-based steering.
    
    Applies orthogonal projection at the specified position to remove bias component.
    
    Args:
        bias_vector: Bias direction vector
        position: Token position to apply projection (-1 for last token)
    
    Returns:
        Hook function
    """
    device = bias_vector.device
    bias_vector = bias_vector.to(device)
    
    def hook_fn(activation, hook):
        # Apply projection at specified position
        if activation.dim() == 3:
            # [batch, seq_len, d_model]
            if position == -1:
                # Project only the last token
                activation[0, -1, :] = orthogonal_projection(
                    activation[0, -1, :].unsqueeze(0),
                    bias_vector
                ).squeeze(0)
            else:
                activation[0, position, :] = orthogonal_projection(
                    activation[0, position, :].unsqueeze(0),
                    bias_vector
                ).squeeze(0)
        elif activation.dim() == 2:
            # [seq_len, d_model]
            if position == -1:
                activation[-1, :] = orthogonal_projection(
                    activation[-1, :].unsqueeze(0),
                    bias_vector
                ).squeeze(0)
            else:
                activation[position, :] = orthogonal_projection(
                    activation[position, :].unsqueeze(0),
                    bias_vector
                ).squeeze(0)
        
        return activation
    
    return hook_fn


def evaluate_projection_steering(
    model: HookedTransformer,
    prompt: str,
    tokenizer: GPT2Tokenizer,
    bias_vector: torch.Tensor,
    layer_idx: int,
    biased_tokens: List[str],
    neutral_tokens: List[str],
    hook_name: str = "hook_resid_post",
    position: int = -1
) -> Dict[str, Any]:
    """
    Evaluate the effect of projection-based steering on bias metrics.
    
    Args:
        model: HookedTransformer model
        prompt: Input text prompt
        tokenizer: Tokenizer instance
        bias_vector: Bias vector to project out
        layer_idx: Layer to apply projection
        biased_tokens: List of stereotypical tokens
        neutral_tokens: List of neutral tokens
        hook_name: Hook point name
        position: Token position to apply projection (-1 for last token)
    
    Returns:
        Dictionary with bias scores (before and after projection)
    """
    device = model.cfg.device
    bias_vector = bias_vector.to(device)
    
    # Get token IDs
    biased_ids = []
    for token in biased_tokens:
        ids = tokenizer.encode(token, add_special_tokens=False)
        biased_ids.extend(ids)
    
    neutral_ids = []
    for token in neutral_tokens:
        ids = tokenizer.encode(token, add_special_tokens=False)
        neutral_ids.extend(ids)
    
    if not biased_ids or not neutral_ids:
        return {}
    
    # Tokenize prompt
    tokens = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    # Baseline: no intervention
    with torch.no_grad():
        logits_baseline = model(tokens)
        next_token_logits_baseline = logits_baseline[0, -1, :]
        log_probs_baseline = F.log_softmax(next_token_logits_baseline, dim=-1)
        
        biased_lp_baseline = torch.logsumexp(log_probs_baseline[biased_ids], dim=0).item()
        neutral_lp_baseline = torch.logsumexp(log_probs_baseline[neutral_ids], dim=0).item()
        bias_score_baseline = biased_lp_baseline - neutral_lp_baseline
    
    # With projection steering
    hook_point = f"blocks.{layer_idx}.{hook_name}"
    projection_hook = projection_steering_hook(bias_vector, position=position)
    
    with torch.no_grad():
        logits_projected = model.run_with_hooks(
            tokens,
            fwd_hooks=[(hook_point, projection_hook)]
        )
        next_token_logits_projected = logits_projected[0, -1, :]
        log_probs_projected = F.log_softmax(next_token_logits_projected, dim=-1)
        
        biased_lp_projected = torch.logsumexp(log_probs_projected[biased_ids], dim=0).item()
        neutral_lp_projected = torch.logsumexp(log_probs_projected[neutral_ids], dim=0).item()
        bias_score_projected = biased_lp_projected - neutral_lp_projected
    
    return {
        "prompt": prompt,
        "layer": layer_idx,
        "bias_score_baseline": bias_score_baseline,
        "bias_score_projected": bias_score_projected,
        "bias_reduction": bias_score_baseline - bias_score_projected,
        "relative_reduction": (bias_score_baseline - bias_score_projected) / (abs(bias_score_baseline) + 1e-10)
    }


def batch_projection_steering_experiment(
    model: HookedTransformer,
    prompts: List[Dict[str, Any]],
    tokenizer: GPT2Tokenizer,
    bias_vectors: Dict[int, torch.Tensor],
    target_layer: int,
    bias_critical_layers: Optional[List[int]] = None,
    hook_name: str = "hook_resid_post",
    position: int = -1
) -> Dict[str, Any]:
    """
    Run projection-based steering experiments on a batch of prompts.
    
    Args:
        model: HookedTransformer model
        prompts: List of prompt dictionaries with 'text', 'biased_tokens', 'neutral_tokens'
        tokenizer: Tokenizer instance
        bias_vectors: Dictionary mapping layer to bias vector
        target_layer: Primary layer to apply projection
        bias_critical_layers: Optional list of layers to test (if None, uses target_layer only)
        hook_name: Hook point name
        position: Token position to apply projection
    
    Returns:
        Aggregated projection steering results
    """
    if target_layer not in bias_vectors:
        raise ValueError(f"Bias vector not found for layer {target_layer}")
    
    # Determine which layers to test
    if bias_critical_layers is None:
        layers_to_test = [target_layer]
    else:
        # Filter to only layers that have bias vectors
        layers_to_test = [l for l in bias_critical_layers if l in bias_vectors]
        if not layers_to_test:
            layers_to_test = [target_layer]
    
    print(f"\n{'='*60}")
    print(f"Batch Projection Steering Experiment")
    print(f"{'='*60}")
    print(f"Prompts: {len(prompts)}")
    print(f"Layers to test: {layers_to_test}")
    print(f"{'='*60}\n")
    
    results = {}
    
    for layer_idx in layers_to_test:
        bias_vector = bias_vectors[layer_idx]
        
        print(f"Testing layer {layer_idx}...")
        
        baseline_scores = []
        projected_scores = []
        reductions = []
        
        for prompt_data in tqdm(prompts, desc=f"Layer {layer_idx}"):
            text = prompt_data.get('text', '')
            biased_tokens = prompt_data.get('biased_tokens', [])
            neutral_tokens = prompt_data.get('neutral_tokens', [])
            
            if not text or not biased_tokens or not neutral_tokens:
                continue
            
            # Evaluate projection steering
            result = evaluate_projection_steering(
                model, text, tokenizer, bias_vector, layer_idx,
                biased_tokens, neutral_tokens, hook_name, position
            )
            
            if result:
                baseline_scores.append(result['bias_score_baseline'])
                projected_scores.append(result['bias_score_projected'])
                reductions.append(result['bias_reduction'])
        
        # Compute statistics
        if baseline_scores:
            results[layer_idx] = {
                "layer": layer_idx,
                "n_prompts": len(baseline_scores),
                "baseline_bias_mean": np.mean(baseline_scores),
                "baseline_bias_std": np.std(baseline_scores),
                "projected_bias_mean": np.mean(projected_scores),
                "projected_bias_std": np.std(projected_scores),
                "mean_reduction": np.mean(reductions),
                "std_reduction": np.std(reductions),
                "relative_reduction": np.mean(reductions) / (abs(np.mean(baseline_scores)) + 1e-10)
            }
    
    return results


def visualize_projection_steering_results(
    results: Dict[int, Dict[str, Any]],
    output_dir: Path = Path("results/projection_steering"),
    save_prefix: str = "projection_steering"
):
    """
    Visualize projection steering experiment results.
    
    Args:
        results: Results from batch_projection_steering_experiment
        output_dir: Directory to save plots
        save_prefix: Prefix for output files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not results:
        print("No results to visualize")
        return
    
    layers = sorted(results.keys())
    
    # Extract data
    baseline_means = [results[l]['baseline_bias_mean'] for l in layers]
    projected_means = [results[l]['projected_bias_mean'] for l in layers]
    reductions = [results[l]['mean_reduction'] for l in layers]
    
    baseline_stds = [results[l]['baseline_bias_std'] for l in layers]
    projected_stds = [results[l]['projected_bias_std'] for l in layers]
    
    # Plot 1: Baseline vs Projected Bias Scores
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Left: Baseline vs Projected
    x = np.arange(len(layers))
    width = 0.35
    
    ax1.bar(x - width/2, baseline_means, width, yerr=baseline_stds, 
            label='Baseline', color='red', alpha=0.7)
    ax1.bar(x + width/2, projected_means, width, yerr=projected_stds,
            label='After Projection', color='green', alpha=0.7)
    
    ax1.axhline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
    ax1.set_xlabel("Layer", fontsize=12)
    ax1.set_ylabel("Bias Score", fontsize=12)
    ax1.set_title("Baseline vs Projected Bias Scores", fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"L{l}" for l in layers])
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Right: Bias Reduction
    colors = ['green' if r > 0 else 'red' for r in reductions]
    ax2.bar(x, reductions, color=colors, alpha=0.7)
    ax2.axhline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
    ax2.set_xlabel("Layer", fontsize=12)
    ax2.set_ylabel("Bias Reduction", fontsize=12)
    ax2.set_title("Bias Reduction by Layer", fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"L{l}" for l in layers])
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / f"{save_prefix}_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    # Save results to JSON
    with open(output_dir / f"{save_prefix}_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Projection steering results saved to {output_dir}")
    
    # Print summary
    print("\n" + "="*60)
    print("Projection Steering Summary")
    print("="*60)
    for layer in layers:
        r = results[layer]
        print(f"\nLayer {layer}:")
        print(f"  Baseline bias: {r['baseline_bias_mean']:.4f} ± {r['baseline_bias_std']:.4f}")
        print(f"  Projected bias: {r['projected_bias_mean']:.4f} ± {r['projected_bias_std']:.4f}")
        print(f"  Reduction: {r['mean_reduction']:.4f} ({r['relative_reduction']*100:.1f}%)")
    print("="*60 + "\n")


def generate_projection_steering_examples(
    model: HookedTransformer,
    prompts: List[str],
    tokenizer: GPT2Tokenizer,
    bias_vector: torch.Tensor,
    layer_idx: int,
    max_new_tokens: int = 20,
    hook_name: str = "hook_resid_post",
    position: int = -1,
    output_file: Optional[Path] = None
) -> List[Dict[str, Any]]:
    """
    Generate example completions with projection-based steering.
    
    Args:
        model: HookedTransformer model
        prompts: List of prompts
        tokenizer: Tokenizer instance
        bias_vector: Bias vector to project out
        layer_idx: Layer to apply projection
        max_new_tokens: Max tokens to generate
        hook_name: Hook point name
        position: Token position to apply projection
        output_file: Optional file to save examples
    
    Returns:
        List of generation examples
    """
    device = model.cfg.device
    bias_vector = bias_vector.to(device)
    
    examples = []
    
    print(f"\n{'='*60}")
    print(f"Generating Projection Steering Examples - Layer {layer_idx}")
    print(f"{'='*60}\n")
    
    hook_point = f"blocks.{layer_idx}.{hook_name}"
    projection_hook = projection_steering_hook(bias_vector, position=position)
    
    for prompt in prompts:
        example = {"prompt": prompt, "completions": {}}
        
        # Baseline (no intervention)
        tokens = tokenizer.encode(prompt, return_tensors="pt").to(device)
        generated_baseline = tokens.clone()
        
        for _ in range(max_new_tokens):
            with torch.no_grad():
                logits = model(generated_baseline)
                next_token_logits = logits[0, -1, :]
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated_baseline = torch.cat([generated_baseline, next_token.unsqueeze(0)], dim=1)
                if next_token.item() == tokenizer.eos_token_id:
                    break
        
        baseline_text = tokenizer.decode(generated_baseline[0], skip_special_tokens=True)
        example["completions"]["baseline"] = baseline_text
        
        # With projection steering
        generated_projected = tokens.clone()
        
        for _ in range(max_new_tokens):
            with torch.no_grad():
                logits = model.run_with_hooks(
                    generated_projected,
                    fwd_hooks=[(hook_point, projection_hook)]
                )
                next_token_logits = logits[0, -1, :]
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated_projected = torch.cat([generated_projected, next_token.unsqueeze(0)], dim=1)
                if next_token.item() == tokenizer.eos_token_id:
                    break
        
        projected_text = tokenizer.decode(generated_projected[0], skip_special_tokens=True)
        example["completions"]["projected"] = projected_text
        
        print(f"Prompt: {prompt}")
        print(f"  Baseline: {baseline_text}")
        print(f"  Projected: {projected_text}")
        print()
        
        examples.append(example)
    
    # Save if requested
    if output_file:
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(examples, f, indent=2)
        print(f"Examples saved to {output_file}")
    
    return examples

