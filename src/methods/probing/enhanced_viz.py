import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import json
import torch
from transformer_lens import HookedTransformer
from transformers import GPT2Tokenizer
from tqdm import tqdm


def plot_probe_roc_curves(
    probe_results: Dict[int, Tuple[Any, Dict[str, float]]],
    X_test_dict: Dict[int, np.ndarray],
    y_test: np.ndarray,
    layers_to_plot: List[int] = None,
    output_dir: Path = Path("results/probing"),
    save_prefix: str = "probe"
):
    """
    Plot ROC curves for selected layers.
    
    Args:
        probe_results: Results from train_layerwise_probes
        X_test_dict: Dictionary mapping layer to test activations
        y_test: Test labels
        layers_to_plot: Specific layers to plot (default: top 5 by AUC)
        output_dir: Output directory
        save_prefix: Prefix for filename
    """
    output_dir = Path(output_dir)
    
    # Select layers to plot
    if layers_to_plot is None:
        # Get top 5 layers by AUC
        sorted_layers = sorted(probe_results.keys(), 
                              key=lambda l: probe_results[l][1]['auc'], 
                              reverse=True)
        layers_to_plot = sorted_layers[:5]
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    for layer in layers_to_plot:
        probe, metrics = probe_results[layer]
        X_test = X_test_dict[layer]
        
        # Get predictions
        y_proba = probe.predict_proba(X_test)[:, 1]
        
        # Compute ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        
        # Plot
        ax.plot(fpr, tpr, linewidth=2, 
               label=f'Layer {layer} (AUC = {roc_auc:.3f})')
    
    # Diagonal reference line
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random')
    
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves: Probe Performance by Layer', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / f"{save_prefix}_roc_curves.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"ROC curves saved to {output_dir}/{save_prefix}_roc_curves.png")


def plot_confusion_matrix(
    probe,
    X_test: np.ndarray,
    y_test: np.ndarray,
    layer_idx: int,
    output_dir: Path = Path("results/probing"),
    save_prefix: str = "probe"
):
    """
    Plot confusion matrix for best probe.
    
    Args:
        probe: Trained LinearProbe
        X_test: Test activations
        y_test: Test labels
        layer_idx: Layer index
        output_dir: Output directory
        save_prefix: Prefix for filename
    """
    output_dir = Path(output_dir)
    
    # Get predictions
    y_pred = probe.predict(X_test)
    
    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
               xticklabels=['Neutral', 'Biased'],
               yticklabels=['Neutral', 'Biased'],
               cbar_kws={'label': 'Count'},
               ax=ax)
    
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)
    ax.set_title(f'Confusion Matrix - Layer {layer_idx}', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / f"{save_prefix}_confusion_matrix_layer{layer_idx}.png", 
               dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"Confusion matrix saved to {output_dir}/{save_prefix}_confusion_matrix_layer{layer_idx}.png")


def plot_head_layer_heatmap(
    biased_heads: List[Tuple[int, int, float]],
    n_layers: int = 24,
    n_heads: int = 16,
    output_dir: Path = Path("results/probing/interventions"),
    save_prefix: str = "interventions"
):
    """
    Create heatmap showing which heads in which layers are most biased.
    
    Args:
        biased_heads: List of (layer, head, contribution) tuples
        n_layers: Total number of layers
        n_heads: Total number of heads per layer
        output_dir: Output directory
        save_prefix: Prefix for filename
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create matrix
    matrix = np.zeros((n_layers, n_heads))
    
    for layer, head, contribution in biased_heads:
        if 0 <= layer < n_layers and 0 <= head < n_heads:
            matrix[layer, head] = abs(contribution)
    
    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    
    sns.heatmap(matrix, cmap='YlOrRd', cbar_kws={'label': 'Bias Contribution'},
               xticklabels=range(n_heads), yticklabels=range(n_layers), ax=ax)
    
    ax.set_xlabel('Head Index', fontsize=12)
    ax.set_ylabel('Layer Index', fontsize=12)
    ax.set_title('Biased Heads Heatmap: Contribution by Layer and Head', 
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / f"{save_prefix}_head_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"Head-layer heatmap saved to {output_dir}/{save_prefix}_head_heatmap.png")


def plot_comparison_dashboard(
    stereoset_results: Dict[str, Any],
    winogender_results: Dict[str, Any],
    output_dir: Path = Path("results/probing"),
    save_prefix: str = "comparison"
):
    """
    Create comparison dashboard between StereoSet and WinoGender results.
    
    Args:
        stereoset_results: Results from StereoSet analysis
        winogender_results: Results from WinoGender analysis
        output_dir: Output directory
        save_prefix: Prefix for filename
    """
    output_dir = Path(output_dir)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Extract data
    st_layers = stereoset_results.get('layers', [])
    st_aucs = stereoset_results.get('aucs', [])
    wg_layers = winogender_results.get('layers', [])
    wg_aucs = winogender_results.get('aucs', [])
    
    # Plot 1: AUC comparison
    axes[0, 0].plot(st_layers, st_aucs, marker='o', linewidth=2, label='StereoSet')
    axes[0, 0].plot(wg_layers, wg_aucs, marker='s', linewidth=2, label='WinoGender')
    axes[0, 0].set_xlabel('Layer', fontsize=11)
    axes[0, 0].set_ylabel('AUC', fontsize=11)
    axes[0, 0].set_title('Probe AUC Comparison', fontsize=12, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Accuracy comparison
    st_accs = stereoset_results.get('accuracies', [])
    wg_accs = winogender_results.get('accuracies', [])
    axes[0, 1].plot(st_layers, st_accs, marker='o', linewidth=2, label='StereoSet')
    axes[0, 1].plot(wg_layers, wg_accs, marker='s', linewidth=2, label='WinoGender')
    axes[0, 1].set_xlabel('Layer', fontsize=11)
    axes[0, 1].set_ylabel('Accuracy', fontsize=11)
    axes[0, 1].set_title('Probe Accuracy Comparison', fontsize=12, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: F1 comparison
    st_f1s = stereoset_results.get('f1s', [])
    wg_f1s = winogender_results.get('f1s', [])
    axes[1, 0].plot(st_layers, st_f1s, marker='o', linewidth=2, label='StereoSet')
    axes[1, 0].plot(wg_layers, wg_f1s, marker='s', linewidth=2, label='WinoGender')
    axes[1, 0].set_xlabel('Layer', fontsize=11)
    axes[1, 0].set_ylabel('F1 Score', fontsize=11)
    axes[1, 0].set_title('Probe F1 Score Comparison', fontsize=12, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Best layer comparison (bar chart)
    st_best = stereoset_results.get('best_layer', {}).get('by_auc', 0)
    wg_best = winogender_results.get('best_layer', {}).get('by_auc', 0)
    
    datasets = ['StereoSet', 'WinoGender']
    best_layers = [st_best, wg_best]
    max_aucs = [max(st_aucs) if st_aucs else 0, max(wg_aucs) if wg_aucs else 0]
    
    x = np.arange(len(datasets))
    width = 0.35
    
    bars1 = axes[1, 1].bar(x - width/2, best_layers, width, label='Best Layer', alpha=0.8)
    ax2 = axes[1, 1].twinx()
    bars2 = ax2.bar(x + width/2, max_aucs, width, label='Max AUC', alpha=0.8, color='orange')
    
    axes[1, 1].set_xlabel('Dataset', fontsize=11)
    axes[1, 1].set_ylabel('Layer Index', fontsize=11, color='C0')
    ax2.set_ylabel('AUC Score', fontsize=11, color='orange')
    axes[1, 1].set_title('Best Performing Layers', fontsize=12, fontweight='bold')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(datasets)
    axes[1, 1].legend(loc='upper left')
    ax2.legend(loc='upper right')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / f"{save_prefix}_dashboard.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"Comparison dashboard saved to {output_dir}/{save_prefix}_dashboard.png")


def create_statistical_summary(
    probe_results: Dict[int, Tuple[Any, Dict[str, float]]],
    output_dir: Path = Path("results/probing"),
    save_prefix: str = "probe"
):
    """
    Create statistical summary with significance tests.
    
    Args:
        probe_results: Results from train_layerwise_probes
        output_dir: Output directory
        save_prefix: Prefix for filename
    """
    output_dir = Path(output_dir)
    
    layers = sorted(probe_results.keys())
    aucs = [probe_results[l][1]['auc'] for l in layers]
    accuracies = [probe_results[l][1]['accuracy'] for l in layers]
    f1s = [probe_results[l][1]['f1'] for l in layers]
    
    # Compute statistics
    summary = {
        "n_layers": len(layers),
        "auc": {
            "mean": float(np.mean(aucs)),
            "std": float(np.std(aucs)),
            "min": float(np.min(aucs)),
            "max": float(np.max(aucs)),
            "median": float(np.median(aucs)),
            "early_layers_mean": float(np.mean(aucs[:8])),  # 0-7
            "mid_layers_mean": float(np.mean(aucs[8:16])),   # 8-15
            "late_layers_mean": float(np.mean(aucs[16:])),   # 16+
        },
        "accuracy": {
            "mean": float(np.mean(accuracies)),
            "std": float(np.std(accuracies)),
            "min": float(np.min(accuracies)),
            "max": float(np.max(accuracies)),
            "median": float(np.median(accuracies)),
        },
        "f1": {
            "mean": float(np.mean(f1s)),
            "std": float(np.std(f1s)),
            "min": float(np.min(f1s)),
            "max": float(np.max(f1s)),
            "median": float(np.median(f1s)),
        },
        "best_layers": {
            "by_auc": int(layers[np.argmax(aucs)]),
            "by_accuracy": int(layers[np.argmax(accuracies)]),
            "by_f1": int(layers[np.argmax(f1s)])
        },
        "layer_groups": {
            "early": {"mean_auc": float(np.mean(aucs[:8]))},
            "mid": {"mean_auc": float(np.mean(aucs[8:16]))},
            "late": {"mean_auc": float(np.mean(aucs[16:]))}
        }
    }
    
    # Save to JSON
    with open(output_dir / f"{save_prefix}_statistical_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    # Create text report
    with open(output_dir / f"{save_prefix}_statistical_summary.txt", "w") as f:
        f.write("="*60 + "\n")
        f.write("Statistical Summary: Linear Probes\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"Number of layers analyzed: {summary['n_layers']}\n\n")
        
        f.write("AUC Statistics:\n")
        f.write(f"  Mean: {summary['auc']['mean']:.4f} ± {summary['auc']['std']:.4f}\n")
        f.write(f"  Range: [{summary['auc']['min']:.4f}, {summary['auc']['max']:.4f}]\n")
        f.write(f"  Median: {summary['auc']['median']:.4f}\n\n")
        
        f.write("AUC by Layer Groups:\n")
        f.write(f"  Early layers (0-7):   {summary['auc']['early_layers_mean']:.4f}\n")
        f.write(f"  Mid layers (8-15):    {summary['auc']['mid_layers_mean']:.4f}\n")
        f.write(f"  Late layers (16+):    {summary['auc']['late_layers_mean']:.4f}\n\n")
        
        f.write("Best Layers:\n")
        f.write(f"  By AUC: Layer {summary['best_layers']['by_auc']}\n")
        f.write(f"  By Accuracy: Layer {summary['best_layers']['by_accuracy']}\n")
        f.write(f"  By F1: Layer {summary['best_layers']['by_f1']}\n")
    
    print(f"Statistical summary saved to {output_dir}/{save_prefix}_statistical_summary.txt")


def plot_metric_heatmap(
    probe_results: Dict[int, Tuple[Any, Dict[str, float]]],
    output_dir: Path = Path("results/probing"),
    save_prefix: str = "probe"
):
    """
    Create heatmap comparing all metrics across layers.
    
    Args:
        probe_results: Results from train_layerwise_probes
        output_dir: Output directory
        save_prefix: Prefix for filename
    """
    output_dir = Path(output_dir)
    
    layers = sorted(probe_results.keys())
    
    # Extract metrics
    metrics_data = {
        'Accuracy': [probe_results[l][1]['accuracy'] for l in layers],
        'AUC': [probe_results[l][1]['auc'] for l in layers],
        'F1 Score': [probe_results[l][1]['f1'] for l in layers]
    }
    
    # Create matrix (metrics x layers)
    matrix = np.array([metrics_data[m] for m in ['Accuracy', 'AUC', 'F1 Score']])
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 4))
    
    sns.heatmap(matrix, annot=False, cmap='RdYlGn', vmin=0, vmax=1,
               xticklabels=layers, yticklabels=['Accuracy', 'AUC', 'F1'],
               cbar_kws={'label': 'Score'}, ax=ax)
    
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Metric', fontsize=12)
    ax.set_title('Probe Performance Heatmap', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / f"{save_prefix}_metric_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"Metric heatmap saved to {output_dir}/{save_prefix}_metric_heatmap.png")


def plot_pareto_frontier(
    intervention_results: Dict[str, Any],
    output_dir: Path = Path("results/probing/interventions"),
    save_prefix: str = "interventions"
):
    """
    Plot Pareto frontier for bias reduction vs side effects.
    
    Shows optimal trade-off points.
    
    Args:
        intervention_results: Results from evaluate_layer_ablation
        output_dir: Output directory
        save_prefix: Prefix for filename
    """
    output_dir = Path(output_dir)
    
    layers = sorted([int(k) for k in intervention_results["layer_results"].keys()])
    bias_reductions = [intervention_results["layer_results"][l]["bias_reduction"] for l in layers]
    ppl_increases = [intervention_results["layer_results"][l]["perplexity_increase"] for l in layers]
    
    # Find Pareto frontier points
    pareto_layers = []
    for i, (br, ppl) in enumerate(zip(bias_reductions, ppl_increases)):
        # A point is Pareto optimal if no other point dominates it
        # (higher bias reduction AND lower ppl increase)
        is_pareto = True
        for j, (br2, ppl2) in enumerate(zip(bias_reductions, ppl_increases)):
            if i != j and br2 >= br and ppl2 <= ppl and (br2 > br or ppl2 < ppl):
                is_pareto = False
                break
        if is_pareto:
            pareto_layers.append(i)
    
    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # All points
    ax.scatter(ppl_increases, bias_reductions, s=100, alpha=0.5, label='All Layers')
    
    # Pareto points
    pareto_ppl = [ppl_increases[i] for i in pareto_layers]
    pareto_br = [bias_reductions[i] for i in pareto_layers]
    ax.scatter(pareto_ppl, pareto_br, s=150, color='red', marker='*', 
              label='Pareto Optimal', zorder=5)
    
    # Connect Pareto points
    sorted_pareto = sorted(zip(pareto_ppl, pareto_br))
    if sorted_pareto:
        pareto_x, pareto_y = zip(*sorted_pareto)
        ax.plot(pareto_x, pareto_y, 'r--', linewidth=2, alpha=0.5)
    
    # Add layer labels
    for i, layer in enumerate(layers):
        ax.annotate(f'L{layer}', (ppl_increases[i], bias_reductions[i]),
                   xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.3)
    ax.axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.3)
    ax.set_xlabel('Perplexity Increase (Side Effect)', fontsize=12)
    ax.set_ylabel('Bias Reduction (Benefit)', fontsize=12)
    ax.set_title('Pareto Frontier: Optimal Bias-PPL Trade-off', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / f"{save_prefix}_pareto_frontier.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"Pareto frontier saved to {output_dir}/{save_prefix}_pareto_frontier.png")
    print(f"Pareto optimal layers: {[layers[i] for i in pareto_layers]}")


def plot_activation_clustering(
    model: HookedTransformer,
    biased_prompts: List[str],
    neutral_prompts: List[str],
    tokenizer: GPT2Tokenizer,
    layer_idx: int,
    method: str = "tsne",
    n_samples: Optional[int] = None,
    output_dir: Path = Path("results/probing"),
    save_prefix: str = "activation_cluster"
):
    """
    Plot t-SNE or PCA visualization of activation space.
    
    Shows whether biased and neutral activations cluster separately,
    validating probe results.
    
    Args:
        model: HookedTransformer model
        biased_prompts: List of biased prompts
        neutral_prompts: List of neutral prompts
        tokenizer: Tokenizer instance
        layer_idx: Layer to extract activations from
        method: "tsne" or "pca"
        n_samples: Number of samples to use (None = all)
        output_dir: Output directory
        save_prefix: Prefix for filename
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nGenerating activation clustering for layer {layer_idx}...")
    
    # Sample prompts if needed
    if n_samples:
        biased_prompts = biased_prompts[:n_samples]
        neutral_prompts = neutral_prompts[:n_samples]
    
    # Collect activations
    device = model.cfg.device
    biased_activations = []
    neutral_activations = []
    
    print("Collecting biased activations...")
    for prompt in tqdm(biased_prompts, desc="Biased"):
        tokens = tokenizer.encode(prompt, return_tensors="pt").to(device)
        cache = {}
        
        def hook_fn(activation, hook):
            cache['activation'] = activation.detach()
            return activation
        
        hook_point = f"blocks.{layer_idx}.hook_resid_post"
        with torch.no_grad():
            model.run_with_hooks(tokens, fwd_hooks=[(hook_point, hook_fn)])
        
        if 'activation' in cache:
            act = cache['activation']
            if act.dim() == 3:
                act = act[0, -1, :]  # Last token
            elif act.dim() == 2:
                act = act[-1, :]
            biased_activations.append(act.detach().cpu().numpy())
    
    print("Collecting neutral activations...")
    for prompt in tqdm(neutral_prompts, desc="Neutral"):
        tokens = tokenizer.encode(prompt, return_tensors="pt").to(device)
        cache = {}
        
        def hook_fn(activation, hook):
            cache['activation'] = activation.detach()
            return activation
        
        hook_point = f"blocks.{layer_idx}.hook_resid_post"
        with torch.no_grad():
            model.run_with_hooks(tokens, fwd_hooks=[(hook_point, hook_fn)])
        
        if 'activation' in cache:
            act = cache['activation']
            if act.dim() == 3:
                act = act[0, -1, :]
            elif act.dim() == 2:
                act = act[-1, :]
            neutral_activations.append(act.detach().cpu().numpy())
    
    # Combine
    all_activations = np.vstack([biased_activations, neutral_activations])
    labels = np.array([1] * len(biased_activations) + [0] * len(neutral_activations))
    
    print(f"Total activations: {len(all_activations)}")
    print(f"  Biased: {len(biased_activations)}")
    print(f"  Neutral: {len(neutral_activations)}")
    
    # Reduce dimensionality
    print(f"\nApplying {method.upper()}...")
    if method.lower() == "tsne":
        reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_activations)-1))
        embedded = reducer.fit_transform(all_activations)
    elif method.lower() == "pca":
        reducer = PCA(n_components=2, random_state=42)
        embedded = reducer.fit_transform(all_activations)
        print(f"Explained variance: {reducer.explained_variance_ratio_.sum():.3f}")
    else:
        raise ValueError(f"Unknown method: {method}. Use 'tsne' or 'pca'")
    
    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    # Plot neutral first (background)
    neutral_mask = labels == 0
    ax.scatter(embedded[neutral_mask, 0], embedded[neutral_mask, 1],
              alpha=0.6, label='Neutral', s=50, color='blue', edgecolors='black', linewidths=0.5)
    
    # Plot biased (foreground)
    biased_mask = labels == 1
    ax.scatter(embedded[biased_mask, 0], embedded[biased_mask, 1],
              alpha=0.6, label='Biased', s=50, color='red', edgecolors='black', linewidths=0.5)
    
    ax.set_xlabel(f'{method.upper()} Component 1', fontsize=12)
    ax.set_ylabel(f'{method.upper()} Component 2', fontsize=12)
    ax.set_title(f'Activation Clustering - Layer {layer_idx} ({method.upper()})', 
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / f"{save_prefix}_layer{layer_idx}_{method}.png", 
               dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"Activation clustering saved to {output_dir}/{save_prefix}_layer{layer_idx}_{method}.png")


def plot_token_probability_heatmap(
    model: HookedTransformer,
    prompts: List[str],
    tokenizer: GPT2Tokenizer,
    target_tokens: List[str],
    output_dir: Path = Path("results/probing"),
    save_prefix: str = "token_probability"
):
    """
    Plot heatmap showing how probability of target tokens changes across layers.
    
    Shows when stereotypical tokens (e.g., "he", "she") become dominant.
    
    Args:
        model: HookedTransformer model
        prompts: List of prompts to analyze
        tokenizer: Tokenizer instance
        target_tokens: List of tokens to track (e.g., ["he", "she"])
        output_dir: Output directory
        save_prefix: Prefix for filename
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nGenerating token probability heatmap...")
    print(f"Target tokens: {target_tokens}")
    print(f"Number of prompts: {len(prompts)}")
    
    # Get token IDs
    token_ids = {}
    for token in target_tokens:
        token_id = tokenizer.encode(token, add_special_tokens=False)
        if len(token_id) == 1:
            token_ids[token] = token_id[0]
        else:
            print(f"Warning: Token '{token}' maps to multiple IDs, skipping")
    
    if not token_ids:
        print("Error: No valid tokens found!")
        return
    
    n_layers = model.cfg.n_layers
    n_tokens = len(token_ids)
    device = model.cfg.device
    
    # Store probabilities: [layer, prompt, token]
    probabilities = np.zeros((n_layers, len(prompts), n_tokens))
    
    print("Computing layer-wise logits...")
    for prompt_idx, prompt in enumerate(tqdm(prompts, desc="Prompts")):
        tokens = tokenizer.encode(prompt, return_tensors="pt").to(device)
        
        # Get logits at each layer using run_with_cache
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens)
        
        for layer_idx in range(n_layers):
            # Get residual stream activation
            resid_key = f"blocks.{layer_idx}.hook_resid_post"
            if resid_key in cache:
                resid_act = cache[resid_key]
                # Get activation at last token position
                if resid_act.dim() == 3:  # [batch, seq, hidden]
                    last_act = resid_act[0, -1, :]  # [hidden]
                elif resid_act.dim() == 2:  # [seq, hidden]
                    last_act = resid_act[-1, :]  # [hidden]
                else:
                    continue
                
                # Project to vocabulary using unembed
                # Add batch dimension if needed
                if last_act.dim() == 1:
                    last_act = last_act.unsqueeze(0)  # [1, hidden]
                
                # Unembed to get logits
                logits = model.unembed(last_act)  # [1, vocab] or [vocab]
                
                # Get logits as 1D tensor
                if logits.dim() == 2:
                    logits = logits[0]  # [vocab]
                
                # Convert to probabilities (detach to avoid gradient issues)
                probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
                
                # Store probabilities for target tokens
                for token_idx, (token, token_id) in enumerate(token_ids.items()):
                    if token_id < len(probs):
                        probabilities[layer_idx, prompt_idx, token_idx] = probs[token_id]
    
    # Average across prompts
    mean_probs = probabilities.mean(axis=1)  # [layers, tokens]
    
    # Create heatmap
    fig, ax = plt.subplots(1, 1, figsize=(10, 12))
    
    # Transpose for better visualization: tokens x layers
    heatmap_data = mean_probs.T
    
    sns.heatmap(heatmap_data, annot=False, cmap='YlOrRd', 
               xticklabels=range(n_layers),
               yticklabels=list(token_ids.keys()),
               cbar_kws={'label': 'Probability'},
               ax=ax)
    
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Token', fontsize=12)
    ax.set_title('Token Probability Heatmap Across Layers', 
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / f"{save_prefix}_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"Token probability heatmap saved to {output_dir}/{save_prefix}_heatmap.png")
    
    # Also create line plot for easier interpretation
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    layers = range(n_layers)
    for token_idx, token in enumerate(token_ids.keys()):
        ax.plot(layers, mean_probs[:, token_idx], marker='o', linewidth=2, 
               label=token, markersize=4)
    
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Mean Probability', fontsize=12)
    ax.set_title('Token Probability Evolution Across Layers', 
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / f"{save_prefix}_evolution.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"Token probability evolution saved to {output_dir}/{save_prefix}_evolution.png")


def plot_attention_flow(
    model: HookedTransformer,
    prompt: str,
    tokenizer: GPT2Tokenizer,
    biased_heads: List[Tuple[int, int]],
    output_dir: Path = Path("results/probing/interventions"),
    save_prefix: str = "attention_flow"
):
    """
    Visualize attention patterns for biased heads.
    
    Shows what tokens the biased heads attend to, providing mechanistic insight.
    
    Args:
        model: HookedTransformer model
        prompt: Example prompt to visualize
        tokenizer: Tokenizer instance
        biased_heads: List of (layer, head) tuples to visualize
        output_dir: Output directory
        save_prefix: Prefix for filename
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nGenerating attention flow diagram...")
    print(f"Prompt: {prompt}")
    print(f"Biased heads: {biased_heads}")
    
    device = model.cfg.device
    tokens = tokenizer.encode(prompt, return_tensors="pt").to(device)
    token_strs = tokenizer.convert_ids_to_tokens(tokens[0])
    
    # Collect attention patterns using run_with_cache
    attention_patterns = {}
    
    try:
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens)
        
        for layer_idx, head_idx in biased_heads:
            attn_key = f"blocks.{layer_idx}.attn.hook_pattern"
            if attn_key in cache:
                attention = cache[attn_key]
                if attention.dim() == 4:  # [batch, heads, seq, seq]
                    attention_patterns[(layer_idx, head_idx)] = attention[0, head_idx, :, :].detach().cpu().numpy()
                elif attention.dim() == 3:  # [heads, seq, seq]
                    attention_patterns[(layer_idx, head_idx)] = attention[head_idx, :, :].detach().cpu().numpy()
                else:
                    print(f"Warning: Unexpected attention shape {attention.shape} for layer {layer_idx}, head {head_idx}")
    except Exception as e:
        print(f"Error extracting attention with cache: {e}")
        # Fallback: try hook-based extraction
        try:
            for layer_idx, head_idx in biased_heads:
                cache_dict = {}
                def attn_hook(activation, hook):
                    cache_dict['attn'] = activation.detach()
                    return activation
                
                hook_point = f"blocks.{layer_idx}.attn.hook_pattern"
                with torch.no_grad():
                    model.run_with_hooks(tokens, fwd_hooks=[(hook_point, attn_hook)])
                
                if 'attn' in cache_dict:
                    attention = cache_dict['attn']
                    if attention.dim() == 4:
                        attention_patterns[(layer_idx, head_idx)] = attention[0, head_idx, :, :].detach().cpu().numpy()
                    elif attention.dim() == 3:
                        attention_patterns[(layer_idx, head_idx)] = attention[head_idx, :, :].detach().cpu().numpy()
        except Exception as e2:
            print(f"Error with hook-based extraction: {e2}")
            return
    
    if not attention_patterns:
        print("Error: Could not extract attention patterns")
        return
    
    # Create visualization
    n_heads = len(attention_patterns)
    fig, axes = plt.subplots(1, n_heads, figsize=(6*n_heads, 6))
    
    if n_heads == 1:
        axes = [axes]
    
    for idx, ((layer_idx, head_idx), attn_matrix) in enumerate(attention_patterns.items()):
        ax = axes[idx]
        
        # Plot attention heatmap
        im = ax.imshow(attn_matrix, cmap='Blues', aspect='auto')
        
        # Set ticks
        ax.set_xticks(range(len(token_strs)))
        ax.set_xticklabels(token_strs, rotation=45, ha='right', fontsize=8)
        ax.set_yticks(range(len(token_strs)))
        ax.set_yticklabels(token_strs, fontsize=8)
        
        ax.set_xlabel('Key (attended to)', fontsize=10)
        ax.set_ylabel('Query (attending from)', fontsize=10)
        ax.set_title(f'Layer {layer_idx}, Head {head_idx}', fontsize=12, fontweight='bold')
        
        # Add colorbar
        plt.colorbar(im, ax=ax, label='Attention Weight')
    
    plt.suptitle('Attention Patterns for Biased Heads', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / f"{save_prefix}.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"Attention flow diagram saved to {output_dir}/{save_prefix}.png")
    
    # Also create a summary plot showing attention to specific tokens
    if len(token_strs) > 0:
        # Find stereotypical tokens (e.g., "he", "she")
        stereotype_tokens = ["he", "she", "his", "her", "him"]
        found_tokens = [t for t in stereotype_tokens if any(t in ts.lower() for ts in token_strs)]
        
        if found_tokens:
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            
            for (layer_idx, head_idx), attn_matrix in attention_patterns.items():
                # Sum attention to stereotype tokens
                token_attention = []
                for token in found_tokens:
                    token_indices = [i for i, ts in enumerate(token_strs) if token in ts.lower()]
                    if token_indices:
                        attn_to_token = attn_matrix[:, token_indices].sum(axis=1)
                        token_attention.append(attn_to_token.mean())
                    else:
                        token_attention.append(0.0)
                
                ax.bar(range(len(found_tokens)), token_attention, 
                      alpha=0.7, label=f'L{layer_idx}H{head_idx}')
            
            ax.set_xticks(range(len(found_tokens)))
            ax.set_xticklabels(found_tokens, fontsize=10)
            ax.set_ylabel('Mean Attention Weight', fontsize=12)
            ax.set_title('Attention to Stereotypical Tokens', fontsize=14, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            plt.savefig(output_dir / f"{save_prefix}_stereotypes.png", dpi=300, bbox_inches="tight")
            plt.close()
            
            print(f"Stereotype attention plot saved to {output_dir}/{save_prefix}_stereotypes.png")

