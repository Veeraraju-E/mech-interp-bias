"""Steering experiments using bias vectors.

Following methodologies from:
- Gupta et al. (2025): Activation steering for bias mitigation
- ICML 2025: No Training Wheels - steering at inference time

Steering modifies activations during generation by adding/subtracting
the bias vector. This should increase (add) or decrease (subtract) bias
in the model's output.
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


def steer_generation(
    model: HookedTransformer,
    prompt: str,
    tokenizer: GPT2Tokenizer,
    bias_vector: torch.Tensor,
    layer_idx: int,
    steering_coeff: float = 1.0,
    hook_name: str = "hook_resid_post",
    max_new_tokens: int = 20,
    temperature: float = 1.0
) -> str:
    """
    Generate text with steering intervention.
    
    During generation, add (steering_coeff * bias_vector) to the specified
    layer's activations. Positive coeff increases bias, negative reduces it.
    
    Args:
        model: HookedTransformer model
        prompt: Input text prompt
        tokenizer: Tokenizer instance
        bias_vector: Bias vector to add/subtract
        layer_idx: Layer to apply steering
        steering_coeff: Coefficient for steering (positive or negative)
        hook_name: Hook point name
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
    
    Returns:
        Generated text
    """
    device = model.cfg.device
    bias_vector = bias_vector.to(device)
    
    # Tokenize prompt
    tokens = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    # Define steering hook
    def steering_hook(activation, hook):
        # Add steering vector to last token position
        if activation.dim() == 3:
            activation[0, -1, :] += steering_coeff * bias_vector
        else:
            activation[-1, :] += steering_coeff * bias_vector
        return activation
    
    hook_point = f"blocks.{layer_idx}.{hook_name}"
    
    # Generate with steering
    generated_tokens = tokens.clone()
    
    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = model.run_with_hooks(
                generated_tokens,
                fwd_hooks=[(hook_point, steering_hook)]
            )
            
            # Get next token logits
            next_token_logits = logits[0, -1, :] / temperature
            
            # Sample next token
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            generated_tokens = torch.cat([generated_tokens, next_token.unsqueeze(0)], dim=1)
            
            # Stop at EOS
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    # Decode
    generated_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
    return generated_text


def evaluate_steering_effect(
    model: HookedTransformer,
    prompt: str,
    tokenizer: GPT2Tokenizer,
    bias_vector: torch.Tensor,
    layer_idx: int,
    biased_tokens: List[str],
    neutral_tokens: List[str],
    steering_coeffs: List[float] = [-2.0, -1.0, 0.0, 1.0, 2.0],
    hook_name: str = "hook_resid_post"
) -> Dict[str, Any]:
    """
    Evaluate the effect of steering on bias metrics.
    
    Measure how bias score changes with different steering coefficients.
    
    Args:
        model: HookedTransformer model
        prompt: Input text prompt
        tokenizer: Tokenizer instance
        bias_vector: Bias vector to apply
        layer_idx: Layer to steer
        biased_tokens: List of stereotypical tokens
        neutral_tokens: List of neutral tokens
        steering_coeffs: List of steering coefficients to test
        hook_name: Hook point name
    
    Returns:
        Dictionary with steering effects
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
    
    results = {
        "prompt": prompt,
        "layer": layer_idx,
        "steering_coeffs": steering_coeffs,
        "bias_scores": []
    }
    
    for coeff in steering_coeffs:
        # Define steering hook
        def steering_hook(activation, hook):
            if activation.dim() == 3:
                activation[0, -1, :] += coeff * bias_vector
            else:
                activation[-1, :] += coeff * bias_vector
            return activation
        
        hook_point = f"blocks.{layer_idx}.{hook_name}"
        
        # Forward pass with steering
        with torch.no_grad():
            logits = model.run_with_hooks(
                tokens,
                fwd_hooks=[(hook_point, steering_hook)]
            )
            
            # Get next token logits
            next_token_logits = logits[0, -1, :]
            log_probs = F.log_softmax(next_token_logits, dim=-1)
            
            # Compute bias score
            biased_lp = torch.logsumexp(log_probs[biased_ids], dim=0).item()
            neutral_lp = torch.logsumexp(log_probs[neutral_ids], dim=0).item()
            bias_score = biased_lp - neutral_lp
            
            results["bias_scores"].append(bias_score)
    
    return results


def batch_steering_experiment(
    model: HookedTransformer,
    prompts: List[Dict[str, Any]],
    tokenizer: GPT2Tokenizer,
    bias_vectors: Dict[int, torch.Tensor],
    target_layer: int,
    steering_coeffs: List[float] = [-2.0, -1.0, 0.0, 1.0, 2.0]
) -> Dict[str, Any]:
    """
    Run steering experiments on a batch of prompts.
    
    Args:
        model: HookedTransformer model
        prompts: List of prompt dictionaries with 'text', 'biased_tokens', 'neutral_tokens'
        tokenizer: Tokenizer instance
        bias_vectors: Dictionary mapping layer to bias vector
        target_layer: Layer to apply steering
        steering_coeffs: Steering coefficients to test
    
    Returns:
        Aggregated steering results
    """
    if target_layer not in bias_vectors:
        raise ValueError(f"Bias vector not found for layer {target_layer}")
    
    bias_vector = bias_vectors[target_layer]
    
    print(f"\n{'='*60}")
    print(f"Batch Steering Experiment - Layer {target_layer}")
    print(f"{'='*60}")
    print(f"Prompts: {len(prompts)}")
    print(f"Steering coefficients: {steering_coeffs}")
    print(f"{'='*60}\n")
    
    # Aggregate results across prompts
    coeff_to_scores = {coeff: [] for coeff in steering_coeffs}
    
    for prompt_data in tqdm(prompts, desc="Running steering experiments"):
        text = prompt_data.get('text', '')
        biased_tokens = prompt_data.get('biased_tokens', [])
        neutral_tokens = prompt_data.get('neutral_tokens', [])
        
        if not text or not biased_tokens or not neutral_tokens:
            continue
        
        # Evaluate steering
        result = evaluate_steering_effect(
            model, text, tokenizer, bias_vector, target_layer,
            biased_tokens, neutral_tokens, steering_coeffs
        )
        
        # Aggregate scores
        for coeff, score in zip(steering_coeffs, result.get("bias_scores", [])):
            coeff_to_scores[coeff].append(score)
    
    # Compute statistics
    summary = {
        "layer": target_layer,
        "steering_coeffs": steering_coeffs,
        "mean_bias_scores": [],
        "std_bias_scores": [],
        "n_prompts": len(prompts)
    }
    
    for coeff in steering_coeffs:
        scores = coeff_to_scores[coeff]
        if scores:
            summary["mean_bias_scores"].append(np.mean(scores))
            summary["std_bias_scores"].append(np.std(scores))
        else:
            summary["mean_bias_scores"].append(0.0)
            summary["std_bias_scores"].append(0.0)
    
    return summary


def visualize_steering_results(
    results: Dict[str, Any],
    output_dir: Path = Path("results/steering"),
    save_prefix: str = "steering"
):
    """
    Visualize steering experiment results.
    
    Args:
        results: Results from batch_steering_experiment
        output_dir: Directory to save plots
        save_prefix: Prefix for output files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    coeffs = results["steering_coeffs"]
    mean_scores = results["mean_bias_scores"]
    std_scores = results["std_bias_scores"]
    
    # Convert to numpy
    mean_scores = np.array(mean_scores)
    std_scores = np.array(std_scores)
    
    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    ax.plot(coeffs, mean_scores, marker='o', linewidth=2, color='purple', label='Mean Bias Score')
    ax.fill_between(
        coeffs,
        mean_scores - std_scores,
        mean_scores + std_scores,
        alpha=0.3,
        color='purple',
        label='±1 Std Dev'
    )
    
    ax.axhline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
    ax.axvline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='No Steering')
    
    ax.set_xlabel("Steering Coefficient", fontsize=12)
    ax.set_ylabel("Bias Score (log prob difference)", fontsize=12)
    ax.set_title(
        f"Steering Effect on Bias Score (Layer {results['layer']})",
        fontsize=14,
        fontweight='bold'
    )
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / f"{save_prefix}_layer{results['layer']}.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    # Save results to JSON
    with open(output_dir / f"{save_prefix}_layer{results['layer']}.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Steering results saved to {output_dir}")
    print(f"Baseline bias (coeff=0): {mean_scores[coeffs.index(0.0)]:.4f}")
    print(f"Effect of negative steering (coeff=-2): {mean_scores[0]:.4f}")
    print(f"Effect of positive steering (coeff=+2): {mean_scores[-1]:.4f}")


def generate_steering_examples(
    model: HookedTransformer,
    prompts: List[str],
    tokenizer: GPT2Tokenizer,
    bias_vector: torch.Tensor,
    layer_idx: int,
    steering_coeffs: List[float] = [-2.0, 0.0, 2.0],
    max_new_tokens: int = 20,
    output_file: Optional[Path] = None
) -> List[Dict[str, Any]]:
    """
    Generate example completions with different steering coefficients.
    
    Args:
        model: HookedTransformer model
        prompts: List of prompts
        tokenizer: Tokenizer instance
        bias_vector: Bias vector to apply
        layer_idx: Layer to steer
        steering_coeffs: Coefficients to test
        max_new_tokens: Max tokens to generate
        output_file: Optional file to save examples
    
    Returns:
        List of generation examples
    """
    examples = []
    
    print(f"\n{'='*60}")
    print(f"Generating Steering Examples - Layer {layer_idx}")
    print(f"{'='*60}\n")
    
    for prompt in prompts:
        example = {"prompt": prompt, "completions": {}}
        
        for coeff in steering_coeffs:
            completion = steer_generation(
                model, prompt, tokenizer, bias_vector, layer_idx,
                steering_coeff=coeff, max_new_tokens=max_new_tokens
            )
            example["completions"][f"coeff_{coeff}"] = completion
            print(f"Prompt: {prompt}")
            print(f"  Coeff={coeff:+.1f}: {completion}")
        
        examples.append(example)
        print()
    
    # Save if requested
    if output_file:
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(examples, f, indent=2)
        print(f"Examples saved to {output_file}")
    
    return examples

