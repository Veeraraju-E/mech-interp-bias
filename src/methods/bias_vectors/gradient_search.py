"""Gradient-based bias vector search (Novel Contribution).

This is a novel method extending prior mean-based steering approaches.
Instead of just computing mean differences, we use gradient ascent to find
an optimal direction in activation space that maximizes the bias metric.

Methodology:
1. Start with a random or mean-difference initialization
2. Apply gradient ascent to maximize bias score when added to activations
3. Compare learned vector with mean-difference vector for validation

This provides a principled, gradient-based way to discover bias directions.
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Callable
from tqdm import tqdm
import matplotlib.pyplot as plt
import json

from transformer_lens import HookedTransformer
from transformers import GPT2Tokenizer


def compute_bias_score_with_vector(
    model: HookedTransformer,
    tokens: torch.Tensor,
    vector: torch.Tensor,
    layer_idx: int,
    biased_token_ids: List[int],
    neutral_token_ids: List[int],
    hook_name: str = "hook_resid_post",
    vector_coeff: float = 1.0
) -> torch.Tensor:
    """
    Compute bias score when adding a vector to activations.
    
    This function is differentiable w.r.t. the vector.
    
    Args:
        model: HookedTransformer model
        tokens: Input tokens
        vector: Vector to add to activations
        layer_idx: Layer to intervene
        biased_token_ids: Token IDs for biased words
        neutral_token_ids: Token IDs for neutral words
        hook_name: Hook point name
        vector_coeff: Coefficient for vector addition
    
    Returns:
        Bias score (differentiable)
    """
    # Storage for modified activation
    cache = {}
    
    def steering_hook(activation, hook):
        if activation.dim() == 3:
            activation[0, -1, :] += vector_coeff * vector
        else:
            activation[-1, :] += vector_coeff * vector
        return activation
    
    hook_point = f"blocks.{layer_idx}.{hook_name}"
    
    # Forward pass with vector addition
    logits = model.run_with_hooks(
        tokens,
        fwd_hooks=[(hook_point, steering_hook)]
    )
    
    # Get next token logits
    next_token_logits = logits[0, -1, :]
    log_probs = F.log_softmax(next_token_logits, dim=-1)
    
    # Compute bias score
    if biased_token_ids and neutral_token_ids:
        biased_ids_tensor = torch.tensor(biased_token_ids, device=log_probs.device, dtype=torch.long)
        neutral_ids_tensor = torch.tensor(neutral_token_ids, device=log_probs.device, dtype=torch.long)
        
        biased_lp = torch.logsumexp(log_probs[biased_ids_tensor], dim=0)
        neutral_lp = torch.logsumexp(log_probs[neutral_ids_tensor], dim=0)
        
        bias_score = biased_lp - neutral_lp
    else:
        bias_score = torch.tensor(0.0, device=log_probs.device)
    
    return bias_score


def gradient_based_vector_search(
    model: HookedTransformer,
    prompts: List[str],
    biased_tokens: List[str],
    neutral_tokens: List[str],
    tokenizer: GPT2Tokenizer,
    layer_idx: int,
    n_steps: int = 100,
    lr: float = 0.01,
    init_vector: Optional[torch.Tensor] = None,
    hook_name: str = "hook_resid_post",
    normalize: bool = True,
    l2_penalty: float = 0.01
) -> Tuple[torch.Tensor, List[float]]:
    """
    Find optimal bias vector using gradient ascent.
    
    This is the novel contribution: instead of just mean difference,
    we optimize a direction that maximally increases bias score.
    
    Args:
        model: HookedTransformer model
        prompts: List of prompts to optimize over
        biased_tokens: Stereotypical tokens
        neutral_tokens: Neutral tokens
        tokenizer: Tokenizer instance
        layer_idx: Layer to search in
        n_steps: Number of optimization steps
        lr: Learning rate
        init_vector: Optional initialization (default: random)
        hook_name: Hook point name
        normalize: Whether to normalize vector at each step
        l2_penalty: L2 regularization penalty
    
    Returns:
        Optimized vector and loss history
    """
    device = model.cfg.device
    hidden_dim = model.cfg.d_model
    
    print(f"\n{'='*60}")
    print(f"Gradient-Based Bias Vector Search - Layer {layer_idx}")
    print(f"{'='*60}")
    print(f"Optimization steps: {n_steps}")
    print(f"Learning rate: {lr}")
    print(f"L2 penalty: {l2_penalty}")
    print(f"Prompts: {len(prompts)}")
    print(f"{'='*60}\n")
    
    # Initialize vector
    if init_vector is not None:
        vector = init_vector.clone().to(device)
    else:
        vector = torch.randn(hidden_dim, device=device) * 0.01
    
    vector.requires_grad = True
    
    # Get token IDs
    biased_ids = []
    for token in biased_tokens:
        ids = tokenizer.encode(token, add_special_tokens=False)
        biased_ids.extend(ids)
    
    neutral_ids = []
    for token in neutral_tokens:
        ids = tokenizer.encode(token, add_special_tokens=False)
        neutral_ids.extend(ids)
    
    # Tokenize all prompts
    prompt_tokens = [
        tokenizer.encode(p, return_tensors="pt").to(device)
        for p in prompts
    ]
    
    # Optimization loop
    optimizer = torch.optim.Adam([vector], lr=lr)
    loss_history = []
    
    for step in tqdm(range(n_steps), desc="Optimizing vector"):
        optimizer.zero_grad()
        
        # Compute bias score across all prompts
        total_bias = 0.0
        for tokens in prompt_tokens:
            bias_score = compute_bias_score_with_vector(
                model, tokens, vector, layer_idx,
                biased_ids, neutral_ids, hook_name
            )
            total_bias += bias_score
        
        # Average bias
        avg_bias = total_bias / len(prompt_tokens)
        
        # Loss = negative bias (we want to maximize bias)
        # Add L2 penalty to prevent unbounded growth
        loss = -avg_bias + l2_penalty * torch.norm(vector)
        
        loss.backward()
        optimizer.step()
        
        # Optionally normalize
        if normalize:
            with torch.no_grad():
                vector.data = vector.data / vector.data.norm()
        
        loss_history.append(loss.item())
        
        if (step + 1) % 10 == 0:
            print(f"Step {step+1}/{n_steps}: Loss={loss.item():.4f}, Bias={avg_bias.item():.4f}, Norm={vector.norm().item():.4f}")
    
    print(f"\n{'='*60}")
    print("Optimization complete!")
    print(f"Final bias score: {-loss_history[-1]:.4f}")
    print(f"Final vector norm: {vector.norm().item():.4f}")
    print(f"{'='*60}\n")
    
    return vector.detach(), loss_history


def optimize_bias_direction(
    model: HookedTransformer,
    biased_prompts: List[str],
    neutral_prompts: List[str],
    tokenizer: GPT2Tokenizer,
    layer_idx: int,
    mean_diff_vector: Optional[torch.Tensor] = None,
    n_steps: int = 100,
    lr: float = 0.01
) -> Dict[str, Any]:
    """
    Comprehensive optimization experiment comparing gradient-based and mean-difference methods.
    
    Args:
        model: HookedTransformer model
        biased_prompts: Biased prompts
        neutral_prompts: Neutral prompts
        tokenizer: Tokenizer instance
        layer_idx: Layer to optimize in
        mean_diff_vector: Pre-computed mean difference vector (for comparison)
        n_steps: Optimization steps
        lr: Learning rate
    
    Returns:
        Dictionary with optimization results and comparison
    """
    # Determine biased vs neutral tokens from prompt content
    # For gender bias, use male vs female pronouns
    biased_tokens = ["he", "him", "his", "man", "male"]
    neutral_tokens = ["she", "her", "hers", "woman", "female"]
    
    # Random initialization
    print("=== Random Initialization ===")
    random_vector, random_loss = gradient_based_vector_search(
        model, biased_prompts + neutral_prompts,
        biased_tokens, neutral_tokens, tokenizer, layer_idx,
        n_steps=n_steps, lr=lr, init_vector=None
    )
    
    # Mean-difference initialization (if provided)
    if mean_diff_vector is not None:
        # Ensure mean_diff_vector is on the correct device
        device = model.cfg.device
        mean_diff_vector = mean_diff_vector.to(device)
        
        print("\n=== Mean-Difference Initialization ===")
        mean_init_vector, mean_init_loss = gradient_based_vector_search(
            model, biased_prompts + neutral_prompts,
            biased_tokens, neutral_tokens, tokenizer, layer_idx,
            n_steps=n_steps, lr=lr, init_vector=mean_diff_vector
        )
        
        # Compute similarity between optimized vectors and mean-diff
        # Ensure all tensors are on the same device
        device = random_vector.device
        mean_diff_vector_device = mean_diff_vector.to(device)
        
        similarity_random = F.cosine_similarity(
            random_vector.unsqueeze(0),
            mean_diff_vector_device.unsqueeze(0)
        ).item()
        
        similarity_mean_init = F.cosine_similarity(
            mean_init_vector.unsqueeze(0),
            mean_diff_vector_device.unsqueeze(0)
        ).item()
    else:
        mean_init_vector = None
        mean_init_loss = []
        similarity_random = 0.0
        similarity_mean_init = 0.0
    
    results = {
        "layer": layer_idx,
        "n_steps": n_steps,
        "learning_rate": lr,
        "random_init": {
            "vector": random_vector.cpu(),
            "loss_history": random_loss,
            "final_loss": random_loss[-1] if random_loss else 0.0,
            "similarity_to_mean_diff": similarity_random
        }
    }
    
    if mean_init_vector is not None:
        results["mean_diff_init"] = {
            "vector": mean_init_vector.cpu(),
            "loss_history": mean_init_loss,
            "final_loss": mean_init_loss[-1] if mean_init_loss else 0.0,
            "similarity_to_mean_diff": similarity_mean_init
        }
    
    return results


def visualize_gradient_search(
    results: Dict[str, Any],
    output_dir: Path = Path("results/gradient_search"),
    save_prefix: str = "gradient_search"
):
    """
    Visualize gradient-based search results.
    
    Args:
        results: Results from optimize_bias_direction
        output_dir: Directory to save plots
        save_prefix: Prefix for output files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot loss curves
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    random_loss = results["random_init"]["loss_history"]
    ax.plot(random_loss, label="Random Init", linewidth=2, color='blue')
    
    if "mean_diff_init" in results:
        mean_loss = results["mean_diff_init"]["loss_history"]
        ax.plot(mean_loss, label="Mean-Diff Init", linewidth=2, color='red')
    
    ax.set_xlabel("Optimization Step", fontsize=12)
    ax.set_ylabel("Loss (negative bias)", fontsize=12)
    ax.set_title(
        f"Gradient-Based Bias Vector Search (Layer {results['layer']})",
        fontsize=14,
        fontweight='bold'
    )
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / f"{save_prefix}_layer{results['layer']}.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    # Save results (excluding tensors for JSON compatibility)
    results_for_json = {
        "layer": results["layer"],
        "n_steps": results["n_steps"],
        "learning_rate": results["learning_rate"],
        "random_init": {
            "loss_history": results["random_init"]["loss_history"],
            "final_loss": results["random_init"]["final_loss"],
            "similarity_to_mean_diff": results["random_init"]["similarity_to_mean_diff"]
        }
    }
    
    if "mean_diff_init" in results:
        results_for_json["mean_diff_init"] = {
            "loss_history": results["mean_diff_init"]["loss_history"],
            "final_loss": results["mean_diff_init"]["final_loss"],
            "similarity_to_mean_diff": results["mean_diff_init"]["similarity_to_mean_diff"]
        }
    
    with open(output_dir / f"{save_prefix}_layer{results['layer']}.json", "w") as f:
        json.dump(results_for_json, f, indent=2)
    
    print(f"Gradient search results saved to {output_dir}")
    print(f"Random init - Final loss: {results['random_init']['final_loss']:.4f}")
    print(f"Random init - Similarity to mean-diff: {results['random_init']['similarity_to_mean_diff']:.4f}")
    
    if "mean_diff_init" in results:
        print(f"Mean-diff init - Final loss: {results['mean_diff_init']['final_loss']:.4f}")
        print(f"Mean-diff init - Similarity to mean-diff: {results['mean_diff_init']['similarity_to_mean_diff']:.4f}")

