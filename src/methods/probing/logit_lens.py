"""Logit Lens analysis for layerwise bias emergence.

Following methodologies from:
- Prakash & Lee (2023): Layered Bias - layerwise decoding of stereotypes
- Nostalgebraist (2020): The Logit Lens technique

The Logit Lens projects hidden states at each layer to vocabulary logits,
allowing us to see at which layer stereotypical tokens begin to dominate.
This reveals how bias emerges progressively through the model.
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import json

from transformer_lens import HookedTransformer
from transformers import GPT2Tokenizer


def apply_logit_lens(
    model: HookedTransformer,
    prompt: str,
    tokenizer: GPT2Tokenizer,
    position: int = -1
) -> Dict[int, torch.Tensor]:
    """
    Apply Logit Lens: project each layer's hidden state to vocabulary logits.
    
    Args:
        model: HookedTransformer model
        prompt: Input text prompt
        tokenizer: Tokenizer instance
        position: Token position to analyze (-1 for last token)
    
    Returns:
        Dictionary mapping layer index to logits over vocabulary
    """
    device = model.cfg.device
    tokens = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    layerwise_logits = {}
    
    # Get all layer outputs
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens)
    
    # Project each layer's residual stream to logits
    n_layers = model.cfg.n_layers
    
    for layer_idx in range(n_layers):
        # Get residual stream at this layer
        hook_name = f"blocks.{layer_idx}.hook_resid_post"
        hidden_state = cache[hook_name]
        
        # Extract position
        if hidden_state.dim() == 3:
            hidden_state = hidden_state[0, position, :]
        else:
            hidden_state = hidden_state[position, :]
        
        # Apply layer norm (final layer norm before unembed)
        if hasattr(model, 'ln_final'):
            hidden_state = model.ln_final(hidden_state)
        
        # Project to vocabulary
        logits = model.unembed(hidden_state)
        layerwise_logits[layer_idx] = logits.cpu()
    
    return layerwise_logits


def compute_layerwise_bias(
    model: HookedTransformer,
    prompt: str,
    biased_tokens: List[str],
    neutral_tokens: List[str],
    tokenizer: GPT2Tokenizer,
    position: int = -1
) -> Dict[int, float]:
    """
    Compute bias score at each layer using Logit Lens.
    
    Bias score = log P(biased tokens) - log P(neutral tokens)
    
    Args:
        model: HookedTransformer model
        prompt: Input text prompt
        biased_tokens: List of stereotypical/biased tokens (e.g., ["he", "him"])
        neutral_tokens: List of neutral/counterstereotypical tokens (e.g., ["she", "her"])
        tokenizer: Tokenizer instance
        position: Token position to analyze
    
    Returns:
        Dictionary mapping layer index to bias score
    """
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
    
    # Get layerwise logits
    layerwise_logits = apply_logit_lens(model, prompt, tokenizer, position)
    
    # Compute bias score at each layer
    bias_scores = {}
    
    for layer_idx, logits in layerwise_logits.items():
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Log prob of biased tokens
        biased_log_probs = log_probs[biased_ids]
        biased_lp = torch.logsumexp(biased_log_probs, dim=0).item()
        
        # Log prob of neutral tokens
        neutral_log_probs = log_probs[neutral_ids]
        neutral_lp = torch.logsumexp(neutral_log_probs, dim=0).item()
        
        # Bias score
        bias_scores[layer_idx] = biased_lp - neutral_lp
    
    return bias_scores


def analyze_bias_emergence(
    model: HookedTransformer,
    prompts: List[Dict[str, Any]],
    tokenizer: GPT2Tokenizer,
    dataset_type: str = "gender"
) -> Dict[str, Any]:
    """
    Analyze how bias emerges across layers for a set of prompts.
    
    Args:
        model: HookedTransformer model
        prompts: List of prompt dictionaries with 'text', 'biased_tokens', 'neutral_tokens'
        tokenizer: Tokenizer instance
        dataset_type: Type of bias ('gender' or 'nationality')
    
    Returns:
        Analysis results including layerwise bias scores
    """
    n_layers = model.cfg.n_layers
    
    # Aggregate bias scores across prompts
    layer_bias_scores = {layer: [] for layer in range(n_layers)}
    
    print(f"\n{'='*60}")
    print(f"Logit Lens Analysis: {dataset_type.capitalize()} Bias Emergence")
    print(f"{'='*60}")
    print(f"Analyzing {len(prompts)} prompts across {n_layers} layers...")
    print(f"{'='*60}\n")
    
    for prompt_data in tqdm(prompts, desc="Processing prompts"):
        text = prompt_data.get('text', '')
        biased_tokens = prompt_data.get('biased_tokens', [])
        neutral_tokens = prompt_data.get('neutral_tokens', [])
        
        if not text or not biased_tokens or not neutral_tokens:
            continue
        
        # Get bias scores for this prompt
        bias_scores = compute_layerwise_bias(
            model, text, biased_tokens, neutral_tokens, tokenizer
        )
        
        # Aggregate
        for layer, score in bias_scores.items():
            layer_bias_scores[layer].append(score)
    
    # Compute statistics
    results = {
        "dataset_type": dataset_type,
        "n_prompts": len(prompts),
        "n_layers": n_layers,
        "layerwise_mean_bias": {},
        "layerwise_std_bias": {},
        "bias_emergence_layer": None
    }
    
    mean_biases = []
    for layer in range(n_layers):
        if layer_bias_scores[layer]:
            mean_bias = np.mean(layer_bias_scores[layer])
            std_bias = np.std(layer_bias_scores[layer])
            results["layerwise_mean_bias"][layer] = float(mean_bias)
            results["layerwise_std_bias"][layer] = float(std_bias)
            mean_biases.append(mean_bias)
        else:
            results["layerwise_mean_bias"][layer] = 0.0
            results["layerwise_std_bias"][layer] = 0.0
            mean_biases.append(0.0)
    
    # Find layer where bias emerges (significant positive bias)
    # Define emergence as first layer where bias > threshold (e.g., 0.5)
    threshold = 0.5
    for layer, bias in enumerate(mean_biases):
        if bias > threshold:
            results["bias_emergence_layer"] = layer
            break
    
    if results["bias_emergence_layer"] is None:
        # Alternatively, use the layer with maximum bias
        results["bias_emergence_layer"] = int(np.argmax(mean_biases))
    
    print(f"\nBias emergence detected at layer: {results['bias_emergence_layer']}")
    print(f"Maximum mean bias: {max(mean_biases):.4f}")
    
    return results


def visualize_logit_lens(
    results: Dict[str, Any],
    output_dir: Path = Path("results/logit_lens"),
    save_prefix: str = "logit_lens"
):
    """
    Visualize Logit Lens analysis results.
    
    Creates plots showing:
    1. Mean bias score by layer
    2. Identification of bias emergence layer
    
    Args:
        results: Results from analyze_bias_emergence
        output_dir: Directory to save plots
        save_prefix: Prefix for output files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    layers = sorted([int(k) for k in results["layerwise_mean_bias"].keys()])
    mean_biases = [results["layerwise_mean_bias"][l] for l in layers]
    std_biases = [results["layerwise_std_bias"][l] for l in layers]
    
    # Convert to numpy arrays
    mean_biases = np.array(mean_biases)
    std_biases = np.array(std_biases)
    
    # Plot bias emergence
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    ax.plot(layers, mean_biases, marker='o', linewidth=2, color='darkred', label='Mean Bias')
    ax.fill_between(
        layers,
        mean_biases - std_biases,
        mean_biases + std_biases,
        alpha=0.3,
        color='red',
        label='±1 Std Dev'
    )
    
    # Mark bias emergence layer
    emergence_layer = results.get("bias_emergence_layer")
    if emergence_layer is not None:
        ax.axvline(
            emergence_layer,
            color='blue',
            linestyle='--',
            linewidth=2,
            label=f'Emergence Layer: {emergence_layer}'
        )
    
    ax.axhline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Bias Score (log prob difference)", fontsize=12)
    ax.set_title(
        f"Logit Lens: Bias Emergence Across Layers\n({results['dataset_type'].capitalize()} Bias)",
        fontsize=14,
        fontweight='bold'
    )
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / f"{save_prefix}_{results['dataset_type']}.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    # Save results to JSON
    with open(output_dir / f"{save_prefix}_{results['dataset_type']}.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Logit Lens results saved to {output_dir}")


def get_top_predicted_tokens(
    model: HookedTransformer,
    prompt: str,
    tokenizer: GPT2Tokenizer,
    layer_idx: int,
    top_k: int = 10,
    position: int = -1
) -> List[Tuple[str, float]]:
    """
    Get top-k predicted tokens at a specific layer using Logit Lens.
    
    Args:
        model: HookedTransformer model
        prompt: Input text prompt
        tokenizer: Tokenizer instance
        layer_idx: Layer to analyze
        top_k: Number of top tokens to return
        position: Token position to analyze
    
    Returns:
        List of (token, probability) tuples
    """
    layerwise_logits = apply_logit_lens(model, prompt, tokenizer, position)
    
    if layer_idx not in layerwise_logits:
        return []
    
    logits = layerwise_logits[layer_idx]
    probs = F.softmax(logits, dim=-1)
    
    # Get top-k
    top_probs, top_indices = torch.topk(probs, k=top_k)
    
    results = []
    for prob, idx in zip(top_probs, top_indices):
        token = tokenizer.decode([idx.item()])
        results.append((token, prob.item()))
    
    return results

