"""Probe-guided interventions for bias mitigation.

Following methodologies from:
- Yang et al. (2025): Head ablation for bias reduction
- Chandna et al. (2025): Component ablation and side-effects
- Yu & Ananiadou (2025): Neuron editing for bias mitigation

This module implements mitigation techniques based on probe findings:
1. Layer ablation: Zero out entire layers identified as bias-encoding
2. Head ablation: Remove specific attention heads in bias layers
3. Attribution-guided ablation: Use attribution patching results to target components
4. Side-effect measurement: Evaluate impact on general capabilities
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Callable
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import json

from transformer_lens import HookedTransformer
from transformers import GPT2Tokenizer


def ablate_layers(
    model: HookedTransformer,
    input_tokens: torch.Tensor,
    layers_to_ablate: List[int],
    hook_name: str = "hook_resid_post"
) -> torch.Tensor:
    """
    Ablate (zero out) specific layers during forward pass.
    
    Args:
        model: HookedTransformer model
        input_tokens: Input token IDs
        layers_to_ablate: List of layer indices to ablate
        hook_name: Hook point name (default: residual stream)
    
    Returns:
        Model output logits with specified layers ablated
    """
    def ablation_hook(activation, hook):
        """Zero out the activation"""
        return torch.zeros_like(activation)
    
    # Create hooks for specified layers
    hooks = [(f"blocks.{layer}.{hook_name}", ablation_hook) 
             for layer in layers_to_ablate]
    
    # Run with hooks
    with torch.no_grad():
        logits = model.run_with_hooks(input_tokens, fwd_hooks=hooks)
    
    return logits


def ablate_heads(
    model: HookedTransformer,
    input_tokens: torch.Tensor,
    heads_to_ablate: List[Tuple[int, int]]
) -> torch.Tensor:
    """
    Ablate specific attention heads during forward pass.
    
    Args:
        model: HookedTransformer model
        input_tokens: Input token IDs
        heads_to_ablate: List of (layer, head) tuples to ablate
    
    Returns:
        Model output logits with specified heads ablated
    """
    # Group heads by layer
    layer_heads = {}
    for layer, head in heads_to_ablate:
        if layer not in layer_heads:
            layer_heads[layer] = []
        layer_heads[layer].append(head)
    
    def create_head_ablation_hook(heads_in_layer):
        """Create hook that ablates specific heads"""
        def hook_fn(activation, hook):
            # activation shape: [batch, pos, head_idx, d_head]
            for head_idx in heads_in_layer:
                activation[:, :, head_idx, :] = 0
            return activation
        return hook_fn
    
    # Create hooks for each layer
    hooks = [(f"blocks.{layer}.attn.hook_z", create_head_ablation_hook(heads))
             for layer, heads in layer_heads.items()]
    
    # Run with hooks
    with torch.no_grad():
        logits = model.run_with_hooks(input_tokens, fwd_hooks=hooks)
    
    return logits


def measure_bias_with_intervention(
    model: HookedTransformer,
    prompts: List[Dict[str, Any]],
    tokenizer: GPT2Tokenizer,
    intervention_fn: Callable,
    dataset_type: str = "gender"
) -> float:
    """
    Measure bias score with a specific intervention applied.
    
    Args:
        model: HookedTransformer model
        prompts: List of prompt dictionaries
        tokenizer: Tokenizer instance
        intervention_fn: Function that modifies model forward pass
        dataset_type: Type of bias to measure
    
    Returns:
        Average bias score
    """
    device = model.cfg.device
    bias_scores = []
    
    for prompt_data in prompts:
        text = prompt_data.get('text', '')
        biased_tokens = prompt_data.get('biased_tokens', [])
        neutral_tokens = prompt_data.get('neutral_tokens', [])
        
        if not text or not biased_tokens or not neutral_tokens:
            continue
        
        # Tokenize
        tokens = tokenizer.encode(text, return_tensors="pt").to(device)
        
        # Get logits with intervention
        logits = intervention_fn(tokens)
        
        # Compute bias score
        next_token_logits = logits[0, -1, :]
        log_probs = F.log_softmax(next_token_logits, dim=-1)
        
        # Get token IDs
        biased_ids = []
        for tok in biased_tokens:
            biased_ids.extend(tokenizer.encode(tok, add_special_tokens=False))
        
        neutral_ids = []
        for tok in neutral_tokens:
            neutral_ids.extend(tokenizer.encode(tok, add_special_tokens=False))
        
        if biased_ids and neutral_ids:
            biased_lp = torch.logsumexp(log_probs[biased_ids], dim=0).item()
            neutral_lp = torch.logsumexp(log_probs[neutral_ids], dim=0).item()
            bias_scores.append(biased_lp - neutral_lp)
    
    return np.mean(bias_scores) if bias_scores else 0.0


def evaluate_layer_ablation(
    model: HookedTransformer,
    bias_prompts: List[Dict[str, Any]],
    general_prompts: List[str],
    tokenizer: GPT2Tokenizer,
    layers_to_test: List[int],
    dataset_type: str = "gender"
) -> Dict[str, Any]:
    """
    Evaluate effect of ablating different layers.
    
    Measures:
    1. Bias reduction
    2. Perplexity change (side effects)
    
    Args:
        model: HookedTransformer model
        bias_prompts: Prompts for bias measurement
        general_prompts: Prompts for perplexity measurement
        tokenizer: Tokenizer instance
        layers_to_test: Layers to try ablating
        dataset_type: Type of bias
    
    Returns:
        Dictionary with results per layer
    """
    device = model.cfg.device
    
    print(f"\n{'='*60}")
    print("Layer Ablation Analysis")
    print(f"{'='*60}")
    print(f"Testing {len(layers_to_test)} layers")
    print(f"Bias prompts: {len(bias_prompts)}")
    print(f"General prompts: {len(general_prompts)}")
    print(f"{'='*60}\n")
    
    # Measure baseline
    print("Measuring baseline (no ablation)...")
    baseline_bias = measure_bias_with_intervention(
        model, bias_prompts, tokenizer,
        lambda tokens: model(tokens),
        dataset_type
    )
    
    baseline_perplexity = compute_perplexity(model, general_prompts, tokenizer)
    
    results = {
        "baseline_bias": baseline_bias,
        "baseline_perplexity": baseline_perplexity,
        "layer_results": {}
    }
    
    # Test each layer
    for layer_idx in tqdm(layers_to_test, desc="Testing layer ablations"):
        # Create intervention function
        def intervention_fn(tokens):
            return ablate_layers(model, tokens, [layer_idx])
        
        # Measure bias with ablation
        ablated_bias = measure_bias_with_intervention(
            model, bias_prompts, tokenizer, intervention_fn, dataset_type
        )
        
        # Measure perplexity change
        ablated_perplexity = compute_perplexity(
            model, general_prompts, tokenizer,
            layers_to_ablate=[layer_idx]
        )
        
        # Compute changes
        bias_reduction = baseline_bias - ablated_bias
        perplexity_increase = ablated_perplexity - baseline_perplexity
        
        results["layer_results"][layer_idx] = {
            "bias_score": ablated_bias,
            "bias_reduction": bias_reduction,
            "perplexity": ablated_perplexity,
            "perplexity_increase": perplexity_increase,
            "effectiveness": bias_reduction / (perplexity_increase + 1e-6)  # Trade-off metric
        }
        
        print(f"Layer {layer_idx}: Bias={ablated_bias:.4f} (Δ{bias_reduction:+.4f}), "
              f"PPL={ablated_perplexity:.2f} (Δ{perplexity_increase:+.2f})")
    
    return results


def find_biased_heads(
    model: HookedTransformer,
    bias_prompts: List[Dict[str, Any]],
    tokenizer: GPT2Tokenizer,
    target_layers: List[int],
    dataset_type: str = "gender",
    top_k: int = 5
) -> List[Tuple[int, int, float]]:
    """
    Identify which attention heads contribute most to bias.
    
    Following Yang et al. (2025) methodology for finding biased heads.
    
    Args:
        model: HookedTransformer model
        bias_prompts: Prompts for bias measurement
        tokenizer: Tokenizer instance
        target_layers: Layers to search in
        dataset_type: Type of bias
        top_k: Number of top biased heads to return
    
    Returns:
        List of (layer, head, bias_contribution) tuples
    """
    print(f"\n{'='*60}")
    print("Finding Biased Attention Heads")
    print(f"{'='*60}")
    print(f"Searching in layers: {target_layers}")
    print(f"Returning top {top_k} heads")
    print(f"{'='*60}\n")
    
    # Measure baseline
    baseline_bias = measure_bias_with_intervention(
        model, bias_prompts, tokenizer,
        lambda tokens: model(tokens),
        dataset_type
    )
    
    head_scores = []
    
    for layer_idx in tqdm(target_layers, desc="Testing heads"):
        for head_idx in range(model.cfg.n_heads):
            # Create intervention that ablates this head
            def intervention_fn(tokens):
                return ablate_heads(model, tokens, [(layer_idx, head_idx)])
            
            # Measure bias with head ablated
            ablated_bias = measure_bias_with_intervention(
                model, bias_prompts, tokenizer, intervention_fn, dataset_type
            )
            
            # Contribution = how much bias decreases when head is removed
            contribution = baseline_bias - ablated_bias
            
            head_scores.append((layer_idx, head_idx, contribution))
    
    # Sort by contribution (descending)
    head_scores.sort(key=lambda x: abs(x[2]), reverse=True)
    
    print("\nTop biased heads:")
    for layer, head, contrib in head_scores[:top_k]:
        print(f"  Layer {layer}, Head {head}: {contrib:+.4f}")
    
    return head_scores[:top_k]


def compute_perplexity(
    model: HookedTransformer,
    prompts: List[str],
    tokenizer: GPT2Tokenizer,
    layers_to_ablate: Optional[List[int]] = None,
    heads_to_ablate: Optional[List[Tuple[int, int]]] = None,
    mlp_layers_to_ablate: Optional[List[int]] = None
) -> float:
    """
    Compute perplexity on a set of prompts.
    
    Used to measure side-effects of interventions.
    
    Args:
        model: HookedTransformer model
        prompts: Text prompts
        tokenizer: Tokenizer instance
        layers_to_ablate: Optional layers to ablate during measurement
    
    Returns:
        Average perplexity
    """
    device = model.cfg.device
    total_loss = 0.0
    total_tokens = 0
    
    for prompt in prompts:
        tokens = tokenizer.encode(prompt, return_tensors="pt").to(device)
        
        if tokens.shape[1] < 2:
            continue
        
        with torch.no_grad():
            if layers_to_ablate:
                logits = ablate_layers(model, tokens, layers_to_ablate)
            elif heads_to_ablate:
                logits = ablate_heads(model, tokens, heads_to_ablate)
            elif mlp_layers_to_ablate:
                hooks = []
                for layer in mlp_layers_to_ablate:
                    hook_name = f"blocks.{layer}.hook_mlp_out"
                    def zero_mlp_hook(activation, hook):
                        return torch.zeros_like(activation)
                    hooks.append((hook_name, zero_mlp_hook))
                logits = model.run_with_hooks(tokens, fwd_hooks=hooks)
            else:
                logits = model(tokens)
        
        # Compute loss
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = tokens[:, 1:].contiguous()
        
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction='sum'
        )
        
        total_loss += loss.item()
        total_tokens += shift_labels.numel()
    
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    perplexity = np.exp(avg_loss)
    
    return perplexity


def load_attribution_results(
    attribution_dir: Path = Path("results/attribution_patching"),
    dataset: str = "stereoset"
) -> Dict[str, Any]:
    """
    Load attribution patching results to guide targeted ablations.
    
    Args:
        attribution_dir: Directory containing attribution results
        dataset: Dataset name
    
    Returns:
        Dictionary with head_scores, mlp_scores, edge_scores
    """
    attribution_dir = Path(attribution_dir)
    results = {}
    
    # Load head ablation results
    head_file = attribution_dir / f"head_ablations_{dataset}.json"
    if head_file.exists():
        with open(head_file, "r") as f:
            head_scores = json.load(f)
        # Convert "layer_head" keys to (layer, head) tuples
        head_dict = {}
        for key, score in head_scores.items():
            layer, head = map(int, key.split("_"))
            head_dict[(layer, head)] = score
        results["head_scores"] = head_dict
        print(f"✅ Loaded {len(head_dict)} head scores from attribution patching")
    
    # Load MLP ablation results
    mlp_file = attribution_dir / f"mlp_ablations_{dataset}.json"
    if mlp_file.exists():
        with open(mlp_file, "r") as f:
            mlp_scores = json.load(f)
        # Convert string keys to int
        mlp_dict = {int(k): v for k, v in mlp_scores.items()}
        results["mlp_scores"] = mlp_dict
        print(f"✅ Loaded {len(mlp_dict)} MLP scores from attribution patching")
    
    # Load edge attribution results
    edge_file = attribution_dir / f"attribution_patching_{dataset}.json"
    if edge_file.exists():
        with open(edge_file, "r") as f:
            edge_scores = json.load(f)
        results["edge_scores"] = edge_scores
        print(f"✅ Loaded edge attribution scores from attribution patching")
    
    return results


def evaluate_attribution_guided_ablation(
    model: HookedTransformer,
    bias_prompts: List[Dict[str, Any]],
    general_prompts: List[str],
    tokenizer: GPT2Tokenizer,
    attribution_results: Dict[str, Any],
    dataset_type: str = "gender",
    top_k_heads: int = 10,
    top_k_mlps: int = 5
) -> Dict[str, Any]:
    """
    Evaluate ablation of top components identified by attribution patching.
    
    This is more targeted than full layer ablation and should have smaller
    side effects while still reducing bias.
    
    Args:
        model: HookedTransformer model
        bias_prompts: Prompts for bias measurement
        general_prompts: Prompts for perplexity measurement
        tokenizer: Tokenizer instance
        attribution_results: Results from load_attribution_results()
        dataset_type: Type of bias
        top_k_heads: Number of top biased heads to ablate
        top_k_mlps: Number of top biased MLPs to ablate
    
    Returns:
        Dictionary with ablation results
    """
    print(f"\n{'='*60}")
    print("Attribution-Guided Targeted Ablation")
    print(f"{'='*60}")
    print(f"Using top {top_k_heads} heads and top {top_k_mlps} MLPs from attribution patching")
    print(f"{'='*60}\n")
    
    # Measure baseline
    baseline_bias = measure_bias_with_intervention(
        model, bias_prompts, tokenizer,
        lambda tokens: model(tokens),
        dataset_type
    )
    baseline_perplexity = compute_perplexity(model, general_prompts, tokenizer)
    
    results = {
        "baseline_bias": baseline_bias,
        "baseline_perplexity": baseline_perplexity,
        "head_ablation": {},
        "mlp_ablation": {},
        "combined_ablation": {}
    }
    
    # Ablate top heads
    if "head_scores" in attribution_results:
        head_scores = attribution_results["head_scores"]
        # Sort by absolute value (most impactful)
        sorted_heads = sorted(head_scores.items(), 
                            key=lambda x: abs(x[1]), 
                            reverse=True)
        top_heads = sorted_heads[:top_k_heads]
        
        print(f"\nAblating top {top_k_heads} heads...")
        heads_to_ablate = [(layer, head) for (layer, head), score in top_heads]
        
        def head_ablation_fn(tokens):
            return ablate_heads(model, tokens, heads_to_ablate)
        
        ablated_bias = measure_bias_with_intervention(
            model, bias_prompts, tokenizer, head_ablation_fn, dataset_type
        )
        
        # Measure perplexity with head ablation
        ablated_perplexity = compute_perplexity(
            model, general_prompts, tokenizer,
            heads_to_ablate=heads_to_ablate
        )
        
        bias_reduction = baseline_bias - ablated_bias
        perplexity_increase = ablated_perplexity - baseline_perplexity
        
        results["head_ablation"] = {
            "heads_ablated": heads_to_ablate,
            "bias_score": ablated_bias,
            "bias_reduction": bias_reduction,
            "perplexity": ablated_perplexity,
            "perplexity_increase": perplexity_increase,
            "effectiveness": bias_reduction / (perplexity_increase + 1e-6)
        }
        
        print(f"Head ablation: Bias={ablated_bias:.4f} (Δ{bias_reduction:+.4f}), "
              f"PPL={ablated_perplexity:.2f} (Δ{perplexity_increase:+.2f})")
    
    # Ablate top MLPs
    if "mlp_scores" in attribution_results:
        mlp_scores = attribution_results["mlp_scores"]
        # Sort by absolute value (most impactful)
        sorted_mlps = sorted(mlp_scores.items(), 
                           key=lambda x: abs(x[1]), 
                           reverse=True)
        top_mlps = sorted_mlps[:top_k_mlps]
        
        print(f"\nAblating top {top_k_mlps} MLPs...")
        mlp_layers = [layer for layer, score in top_mlps]
        
        def mlp_ablation_fn(tokens):
            hooks = []
            for layer in mlp_layers:
                hook_name = f"blocks.{layer}.hook_mlp_out"
                def zero_mlp_hook(activation, hook):
                    return torch.zeros_like(activation)
                hooks.append((hook_name, zero_mlp_hook))
            
            with torch.no_grad():
                return model.run_with_hooks(tokens, fwd_hooks=hooks)
        
        ablated_bias = measure_bias_with_intervention(
            model, bias_prompts, tokenizer, mlp_ablation_fn, dataset_type
        )
        
        # Measure perplexity with MLP ablation
        ablated_perplexity = compute_perplexity(
            model, general_prompts, tokenizer,
            mlp_layers_to_ablate=mlp_layers
        )
        
        bias_reduction = baseline_bias - ablated_bias
        perplexity_increase = ablated_perplexity - baseline_perplexity
        
        results["mlp_ablation"] = {
            "mlps_ablated": mlp_layers,
            "bias_score": ablated_bias,
            "bias_reduction": bias_reduction,
            "perplexity": ablated_perplexity,
            "perplexity_increase": perplexity_increase,
            "effectiveness": bias_reduction / (perplexity_increase + 1e-6)
        }
        
        print(f"MLP ablation: Bias={ablated_bias:.4f} (Δ{bias_reduction:+.4f}), "
              f"PPL={ablated_perplexity:.2f} (Δ{perplexity_increase:+.2f})")
    
    # Combined ablation (top heads + top MLPs)
    if "head_scores" in attribution_results and "mlp_scores" in attribution_results:
        print(f"\nAblating top {top_k_heads} heads + top {top_k_mlps} MLPs...")
        
        def combined_ablation_fn(tokens):
            hooks = []
            
            # Add head ablation hooks
            for layer, head in heads_to_ablate:
                hook_name = f"blocks.{layer}.attn.hook_z"
                def create_head_hook(h):
                    def head_hook(activation, hook):
                        act = activation.clone()
                        act[:, :, h, :] = 0.0
                        return act
                    return head_hook
                hooks.append((hook_name, create_head_hook(head)))
            
            # Add MLP ablation hooks
            for layer in mlp_layers:
                hook_name = f"blocks.{layer}.hook_mlp_out"
                def zero_mlp_hook(activation, hook):
                    return torch.zeros_like(activation)
                hooks.append((hook_name, zero_mlp_hook))
            
            with torch.no_grad():
                return model.run_with_hooks(tokens, fwd_hooks=hooks)
        
        ablated_bias = measure_bias_with_intervention(
            model, bias_prompts, tokenizer, combined_ablation_fn, dataset_type
        )
        
        # For combined ablation, compute perplexity manually
        device = model.cfg.device
        total_loss = 0.0
        total_tokens = 0
        
        for prompt in general_prompts:
            tokens = tokenizer.encode(prompt, return_tensors="pt").to(device)
            if tokens.shape[1] < 2:
                continue
            
            with torch.no_grad():
                logits = combined_ablation_fn(tokens)
            
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = tokens[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction='sum'
            )
            total_loss += loss.item()
            total_tokens += shift_labels.numel()
        
        avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
        ablated_perplexity = np.exp(avg_loss)
        
        bias_reduction = baseline_bias - ablated_bias
        perplexity_increase = ablated_perplexity - baseline_perplexity
        
        results["combined_ablation"] = {
            "heads_ablated": heads_to_ablate,
            "mlps_ablated": mlp_layers,
            "bias_score": ablated_bias,
            "bias_reduction": bias_reduction,
            "perplexity": ablated_perplexity,
            "perplexity_increase": perplexity_increase,
            "effectiveness": bias_reduction / (perplexity_increase + 1e-6)
        }
        
        print(f"Combined ablation: Bias={ablated_bias:.4f} (Δ{bias_reduction:+.4f}), "
              f"PPL={ablated_perplexity:.2f} (Δ{perplexity_increase:+.2f})")
    
    return results


def visualize_intervention_results(
    results: Dict[str, Any],
    output_dir: Path = Path("results/probing/interventions"),
    save_prefix: str = "interventions"
):
    """
    Visualize results of probe-guided interventions.
    
    Creates plots showing:
    1. Bias reduction vs side-effects trade-off
    2. Per-layer effectiveness
    
    Args:
        results: Results from evaluate_layer_ablation
        output_dir: Directory to save plots
        save_prefix: Prefix for output files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    layers = sorted([int(k) for k in results["layer_results"].keys()])
    bias_reductions = [results["layer_results"][l]["bias_reduction"] for l in layers]
    perplexity_increases = [results["layer_results"][l]["perplexity_increase"] for l in layers]
    effectiveness = [results["layer_results"][l]["effectiveness"] for l in layers]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Left plot: Trade-off scatter
    scatter = ax1.scatter(perplexity_increases, bias_reductions, 
                         c=layers, cmap='viridis', s=100, alpha=0.7)
    
    # Add layer labels
    for i, layer in enumerate(layers):
        ax1.annotate(f'L{layer}', 
                    (perplexity_increases[i], bias_reductions[i]),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    ax1.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax1.axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax1.set_xlabel("Perplexity Increase (Side Effect)", fontsize=12)
    ax1.set_ylabel("Bias Reduction", fontsize=12)
    ax1.set_title("Trade-off: Bias Reduction vs Side Effects", fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax1, label='Layer')
    
    # Right plot: Effectiveness by layer
    colors = ['green' if e > 0 else 'red' for e in effectiveness]
    ax2.bar(layers, effectiveness, color=colors, alpha=0.7)
    ax2.axhline(0, color='black', linestyle='-', linewidth=1)
    ax2.set_xlabel("Layer", fontsize=12)
    ax2.set_ylabel("Effectiveness (Bias Reduction / PPL Increase)", fontsize=12)
    ax2.set_title("Layer Ablation Effectiveness", fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / f"{save_prefix}_layer_ablation.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    # Save results to JSON
    results_for_json = {
        "baseline_bias": results["baseline_bias"],
        "baseline_perplexity": results["baseline_perplexity"],
        "layers": layers,
        "bias_reductions": bias_reductions,
        "perplexity_increases": perplexity_increases,
        "effectiveness": effectiveness
    }
    
    with open(output_dir / f"{save_prefix}_results.json", "w") as f:
        json.dump(results_for_json, f, indent=2)
    
    print(f"Intervention results saved to {output_dir}")
    
    # Print summary
    best_idx = np.argmax(effectiveness)
    print(f"\nBest layer to ablate: {layers[best_idx]}")
    print(f"  Bias reduction: {bias_reductions[best_idx]:.4f}")
    print(f"  Perplexity increase: {perplexity_increases[best_idx]:.2f}")
    print(f"  Effectiveness: {effectiveness[best_idx]:.4f}")

