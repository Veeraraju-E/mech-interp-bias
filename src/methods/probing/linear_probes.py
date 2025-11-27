"""Linear probes for detecting bias in hidden-layer activations.

Following methodologies from:
- Nanda et al. (2024): Probing for linear separability of concepts
- Prakash & Lee (2023): Layered Bias analysis
- Gupta et al. (2025): Activation steering with probes

This module trains simple logistic regression classifiers on hidden activations
to predict whether a prompt is "biased" or "neutral". High probe accuracy at a
given layer indicates that bias is linearly decodable at that layer.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import json

from transformer_lens import HookedTransformer
from transformers import GPT2Tokenizer


class LinearProbe:
    """
    Simple linear probe (logistic regression) for bias detection.
    
    Trained on activation vectors from a specific layer to predict
    if an input is biased or neutral.
    """
    
    def __init__(self, input_dim: int, max_iter: int = 1000, random_state: int = 42):
        """
        Initialize linear probe.
        
        Args:
            input_dim: Dimensionality of input activations
            max_iter: Maximum iterations for logistic regression
            random_state: Random seed
        """
        self.input_dim = input_dim
        self.max_iter = max_iter
        self.random_state = random_state
        self.model = LogisticRegression(
            max_iter=max_iter,
            random_state=random_state,
            solver='lbfgs'
        )
        self.is_trained = False
    
    def train(self, X: np.ndarray, y: np.ndarray):
        """
        Train the probe.
        
        Args:
            X: Activations, shape (n_samples, input_dim)
            y: Binary labels (0=neutral, 1=biased)
        """
        self.model.fit(X, y)
        self.is_trained = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Get binary predictions."""
        if not self.is_trained:
            raise ValueError("Probe must be trained before prediction")
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get probability predictions."""
        if not self.is_trained:
            raise ValueError("Probe must be trained before prediction")
        return self.model.predict_proba(X)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate probe performance.
        
        Returns:
            Dictionary with accuracy, AUC, and F1 scores
        """
        if not self.is_trained:
            raise ValueError("Probe must be trained before evaluation")
        
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)[:, 1]
        
        return {
            "accuracy": accuracy_score(y, y_pred),
            "auc": roc_auc_score(y, y_proba),
            "f1": f1_score(y, y_pred)
        }
    
    def get_weights(self) -> np.ndarray:
        """Get probe weight vector (bias direction)."""
        if not self.is_trained:
            raise ValueError("Probe must be trained to extract weights")
        return self.model.coef_[0]


def collect_layer_activations(
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
        hook_name: Name of hook point (default: residual stream post-attention)
        position: Token position to extract (-1 for last token)
    
    Returns:
        Tensor of activations, shape (n_prompts, hidden_dim)
    """
    device = model.cfg.device
    activations = []
    
    for prompt in tqdm(prompts, desc=f"Collecting layer {layer_idx} activations"):
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


def train_layerwise_probes(
    model: HookedTransformer,
    biased_prompts: List[str],
    neutral_prompts: List[str],
    tokenizer: GPT2Tokenizer,
    test_size: float = 0.2,
    hook_name: str = "hook_resid_post",
    random_state: int = 42
) -> Dict[int, Tuple[LinearProbe, Dict[str, float]]]:
    """
    Train linear probes for all layers of the model.
    
    This implements the methodology from Section 4.2: train a logistic regression
    on each layer's residual stream to predict if input is biased or neutral.
    
    Args:
        model: HookedTransformer model
        biased_prompts: List of biased text prompts
        neutral_prompts: List of neutral text prompts
        tokenizer: Tokenizer instance
        test_size: Fraction of data for testing
        hook_name: Hook point name (default: residual stream)
        random_state: Random seed
    
    Returns:
        Dictionary mapping layer index to (probe, evaluation_metrics)
    """
    n_layers = model.cfg.n_layers
    results = {}
    
    print(f"\n{'='*60}")
    print("Training Layerwise Linear Probes")
    print(f"{'='*60}")
    print(f"Model: {model.cfg.model_name}")
    print(f"Layers: {n_layers}")
    print(f"Biased prompts: {len(biased_prompts)}")
    print(f"Neutral prompts: {len(neutral_prompts)}")
    print(f"{'='*60}\n")
    
    # Create labels (1 for biased, 0 for neutral)
    all_prompts = biased_prompts + neutral_prompts
    labels = np.array([1] * len(biased_prompts) + [0] * len(neutral_prompts))
    
    # Split into train/test
    prompts_train, prompts_test, y_train, y_test = train_test_split(
        all_prompts, labels, test_size=test_size, random_state=random_state, stratify=labels
    )
    
    for layer_idx in range(n_layers):
        print(f"\n--- Layer {layer_idx} ---")
        
        # Collect training activations
        X_train = collect_layer_activations(
            model, prompts_train, tokenizer, layer_idx, hook_name
        )
        X_train = X_train.numpy()
        
        # Collect test activations
        X_test = collect_layer_activations(
            model, prompts_test, tokenizer, layer_idx, hook_name
        )
        X_test = X_test.numpy()
        
        # Train probe
        probe = LinearProbe(input_dim=X_train.shape[1], random_state=random_state)
        probe.train(X_train, y_train)
        
        # Evaluate
        metrics = probe.evaluate(X_test, y_test)
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"AUC: {metrics['auc']:.4f}")
        print(f"F1: {metrics['f1']:.4f}")
        
        results[layer_idx] = (probe, metrics)
    
    print(f"\n{'='*60}")
    print("Probe training complete!")
    print(f"{'='*60}\n")
    
    return results


def evaluate_probe(
    probe: LinearProbe,
    X: np.ndarray,
    y: np.ndarray
) -> Dict[str, float]:
    """Convenience function to evaluate a trained probe."""
    return probe.evaluate(X, y)


def plot_probe_results(
    results: Dict[int, Tuple[LinearProbe, Dict[str, float]]],
    output_dir: Path = Path("results/probing"),
    save_prefix: str = "probe"
):
    """
    Visualize probe results across layers.
    
    Creates plots showing:
    1. Accuracy/AUC/F1 by layer
    2. Identification of layers where bias becomes linearly separable
    
    Args:
        results: Dictionary from train_layerwise_probes
        output_dir: Directory to save plots
        save_prefix: Prefix for output files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    layers = sorted(results.keys())
    accuracies = [results[l][1]['accuracy'] for l in layers]
    aucs = [results[l][1]['auc'] for l in layers]
    f1s = [results[l][1]['f1'] for l in layers]
    
    # Plot all metrics
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    ax.plot(layers, accuracies, marker='o', label='Accuracy', linewidth=2)
    ax.plot(layers, aucs, marker='s', label='AUC', linewidth=2)
    ax.plot(layers, f1s, marker='^', label='F1', linewidth=2)
    
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Linear Probe Performance by Layer", fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    plt.savefig(output_dir / f"{save_prefix}_layerwise.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    # Save results to JSON
    results_dict = {
        "layers": layers,
        "accuracies": accuracies,
        "aucs": aucs,
        "f1s": f1s,
        "best_layer": {
            "by_auc": int(layers[np.argmax(aucs)]),
            "by_accuracy": int(layers[np.argmax(accuracies)]),
            "by_f1": int(layers[np.argmax(f1s)])
        }
    }
    
    with open(output_dir / f"{save_prefix}_results.json", "w") as f:
        json.dump(results_dict, f, indent=2)
    
    print(f"Probe results saved to {output_dir}")
    print(f"Best layer by AUC: {results_dict['best_layer']['by_auc']} (AUC={max(aucs):.4f})")

