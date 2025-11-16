"""Analysis and visualization scripts for bias activation patching results."""

import json
import csv
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def load_results(results_dir: Path = Path("results")) -> Dict:
    """Load all experiment results."""
    summary_file = results_dir / "summary.json"
    if summary_file.exists():
        with open(summary_file, "r") as f:
            return json.load(f)
    return {}


def plot_edge_impacts(
    impact_scores: Dict[str, float],
    title: str,
    output_file: Path,
    top_n: int = 20
):
    """Plot top N edge impacts."""
    ranked = sorted(impact_scores.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n]
    
    hook_names = [name.split(".")[-1] if "." in name else name for name, _ in ranked]
    scores = [score for _, score in ranked]
    
    plt.figure(figsize=(12, 6))
    plt.barh(range(len(hook_names)), scores)
    plt.yticks(range(len(hook_names)), hook_names)
    plt.xlabel("Impact Score")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()


def plot_layer_impacts(
    impact_scores: Dict[str, float],
    title: str,
    output_file: Path
):
    """Plot impact scores aggregated by layer."""
    layer_scores = {}
    
    for hook_name, score in impact_scores.items():
        # Extract layer number from hook name
        parts = hook_name.split(".")
        if len(parts) >= 2 and parts[0] == "blocks":
            try:
                layer = int(parts[1])
                if layer not in layer_scores:
                    layer_scores[layer] = []
                layer_scores[layer].append(abs(score))
            except ValueError:
                pass
    
    # Average scores per layer
    layer_avgs = {layer: np.mean(scores) for layer, scores in layer_scores.items()}
    
    layers = sorted(layer_avgs.keys())
    scores = [layer_avgs[l] for l in layers]
    
    plt.figure(figsize=(10, 6))
    plt.bar(layers, scores)
    plt.xlabel("Layer")
    plt.ylabel("Average Impact Score")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()


def plot_head_heatmap(
    head_impacts: Dict[Tuple[int, int], float],
    title: str,
    output_file: Path,
    n_layers: int = 24,
    n_heads: int = 16
):
    """Plot heatmap of head impacts."""
    # Create matrix
    matrix = np.zeros((n_layers, n_heads))
    
    for (layer, head), impact in head_impacts.items():
        if 0 <= layer < n_layers and 0 <= head < n_heads:
            matrix[layer, head] = abs(impact)
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(matrix, cmap="YlOrRd", cbar_kws={"label": "Impact Score"})
    plt.xlabel("Head")
    plt.ylabel("Layer")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()


def generate_summary_report(results_dir: Path = Path("results")):
    """Generate a text summary report of results."""
    results = load_results(results_dir)
    
    if not results:
        print("No results found. Run experiments first.")
        return
    
    report_file = results_dir / "summary_report.txt"
    
    with open(report_file, "w") as f:
        f.write("="*60 + "\n")
        f.write("Bias Activation Patching - Summary Report\n")
        f.write("="*60 + "\n\n")
        
        for dataset_name, dataset_results in results.items():
            f.write(f"\nDataset: {dataset_name.upper()}\n")
            f.write("-"*60 + "\n")
            
            baseline = dataset_results.get("baseline_bias", 0.0)
            f.write(f"Baseline Bias Score: {baseline:.4f}\n\n")
            
            # Causal patching results
            causal = dataset_results.get("causal_patching", {})
            if causal and "ranked_edges" in causal:
                f.write("Top 5 Causal Patching Edges:\n")
                for hook_name, score in causal["ranked_edges"][:5]:
                    f.write(f"  {hook_name}: {score:.4f}\n")
                f.write("\n")
            
            # Attribution patching results
            attribution = dataset_results.get("attribution_patching", {})
            if attribution and "ranked_edges" in attribution:
                f.write("Top 5 Attribution Patching Edges:\n")
                for hook_name, score in attribution["ranked_edges"][:5]:
                    f.write(f"  {hook_name}: {score:.4f}\n")
                f.write("\n")
            
            # Ablation results
            ablations = dataset_results.get("ablations", {})
            if ablations and "ranked_heads" in ablations:
                f.write("Top 5 Biased Heads:\n")
                for (layer, head), impact in ablations["ranked_heads"][:5]:
                    f.write(f"  Layer {layer}, Head {head}: {impact:.4f}\n")
                f.write("\n")
            
            if ablations and "ranked_mlps" in ablations:
                f.write("Top 5 Biased MLPs:\n")
                for layer, impact in ablations["ranked_mlps"][:5]:
                    f.write(f"  Layer {layer}: {impact:.4f}\n")
                f.write("\n")
    
    print(f"Summary report saved to {report_file}")


def create_visualizations(results_dir: Path = Path("results")):
    """Create all visualizations from results."""
    results_dir = Path(results_dir)
    viz_dir = results_dir / "visualizations"
    viz_dir.mkdir(exist_ok=True)
    
    results = load_results(results_dir)
    
    if not results:
        print("No results found. Run experiments first.")
        return
    
    for dataset_name, dataset_results in results.items():
        # Causal patching visualizations
        causal = dataset_results.get("causal_patching", {})
        if causal and "impact_scores" in causal:
            plot_edge_impacts(
                causal["impact_scores"],
                f"Top Edge Impacts - Causal Patching ({dataset_name})",
                viz_dir / f"causal_edges_{dataset_name}.png"
            )
            plot_layer_impacts(
                causal["impact_scores"],
                f"Layer Impacts - Causal Patching ({dataset_name})",
                viz_dir / f"causal_layers_{dataset_name}.png"
            )
        
        # Attribution patching visualizations
        attribution = dataset_results.get("attribution_patching", {})
        if attribution and "attributions" in attribution:
            plot_edge_impacts(
                attribution["attributions"],
                f"Top Edge Attributions ({dataset_name})",
                viz_dir / f"attribution_edges_{dataset_name}.png"
            )
            plot_layer_impacts(
                attribution["attributions"],
                f"Layer Attributions ({dataset_name})",
                viz_dir / f"attribution_layers_{dataset_name}.png"
            )
        
        # Ablation visualizations
        ablations = dataset_results.get("ablations", {})
        if ablations and "head_impacts" in ablations:
            # Convert tuple keys back
            head_dict = {}
            for key, value in ablations["head_impacts"].items():
                if isinstance(key, str) and "_" in key:
                    layer, head = map(int, key.split("_"))
                    head_dict[(layer, head)] = value
                elif isinstance(key, list) and len(key) == 2:
                    head_dict[tuple(key)] = value
            
            if head_dict:
                plot_head_heatmap(
                    head_dict,
                    f"Head Impact Heatmap ({dataset_name})",
                    viz_dir / f"head_heatmap_{dataset_name}.png"
                )
    
    print(f"Visualizations saved to {viz_dir}")


def main():
    """Main analysis function."""
    results_dir = Path("results")
    
    print("Generating summary report...")
    generate_summary_report(results_dir)
    
    print("Creating visualizations...")
    create_visualizations(results_dir)
    
    print("Analysis complete!")


if __name__ == "__main__":
    main()

