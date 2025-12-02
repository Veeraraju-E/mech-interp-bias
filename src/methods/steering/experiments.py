import copy
import json
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import torch
import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table
from rich.panel import Panel

from src.model_setup import get_tokenizer, load_model, setup_device
from .dataset import collect_activation_dataset
from .evaluation import evaluate_bias_reduction, evaluate_perplexity
from .feature_analysis import compute_latent_correlations, select_bias_latents
from .mmlu import DEFAULT_MMLU_SUBJECTS, evaluate_mmlu_accuracy, load_mmlu_samples
from .suppression import create_suppression_hook
from .train_sae import train_sae
from .visualization import plot_activation_profile, plot_latent_profile

app = typer.Typer(help="SAE steering experiment (Hegde 2024)")
console = Console()


def _create_activation_histogram(model, tokenizer, entries, layer, output_dir):
    """Create and save activation histogram if sufficient prompts available."""
    biased_prompts = [e["prompt"] for e in entries if e.get("male_prob", 0.0) > e.get("female_prob", 0.0)]
    neutral_prompts = [e["prompt"] for e in entries if e.get("male_prob", 0.0) <= e.get("female_prob", 0.0)]
    if not (biased_prompts and neutral_prompts):
        console.print("[yellow]Skipping activation histogram (insufficient biased/neutral prompts).[/yellow]")
        return None

    hist_dir = output_dir / "visualizations"
    hist_dir.mkdir(parents=True, exist_ok=True)
    hist_path = hist_dir / f"activation_profile_layer{layer}.png"
    raw_profile = plot_activation_profile(model, tokenizer, biased_prompts, neutral_prompts, layer, hist_path)
    profile_data = {
        "biased_means": raw_profile["biased_means"].tolist(), "neutral_means": raw_profile["neutral_means"].tolist(),
        "num_neurons": raw_profile["num_neurons"], "num_biased_prompts": raw_profile["num_biased_prompts"],
        "num_neutral_prompts": raw_profile["num_neutral_prompts"],
    }
    (output_dir / "activation_profiles").mkdir(parents=True, exist_ok=True)
    with open(output_dir / "activation_profiles" / f"layer{layer}.json", "w") as f:
        json.dump(profile_data, f, indent=2)
    console.print(f"[green]✓[/green] Saved activation profile plot to {hist_path}")
    return profile_data


def _load_mmlu_baseline(model, tokenizer, mmlu_subjects, mmlu_max_questions):
    """Load MMLU samples and compute baseline metrics."""
    effective_subjects = list(mmlu_subjects) if mmlu_subjects else list(DEFAULT_MMLU_SUBJECTS)
    console.print(f"[cyan]Loading MMLU subjects[/cyan] {effective_subjects} ([dim]<= {mmlu_max_questions} questions per subject[/dim])")
    mmlu_samples = load_mmlu_samples(effective_subjects, max_questions_per_subject=mmlu_max_questions)
    if not mmlu_samples:
        console.print("[yellow]Warning:[/yellow] No MMLU samples were loaded; skipping MMLU evaluation.")
        return [], None, None
    
    baseline_metrics = evaluate_mmlu_accuracy(model, tokenizer, mmlu_samples)
    console.print(f"[green]Baseline MMLU accuracy:[/green] {baseline_metrics['overall_accuracy']:.3f} over {baseline_metrics['num_questions']} questions")
    return mmlu_samples, baseline_metrics, effective_subjects


def _compute_latent_profile(sae, activations_tensor, bias_mask, layer, k_sparse, output_dir):
    """Compute and save latent profile visualization."""
    with torch.no_grad():
        latents = torch.cat([sae.encode(activations_tensor[start:start+512], apply_sparsity=True).cpu()
                             for start in range(0, activations_tensor.shape[0], 512)], dim=0).numpy()
    
    if latents.shape[0] != bias_mask.shape[0] or not (latents[bias_mask].size and latents[~bias_mask].size):
        return None

    latent_plot_dir = output_dir / "latent_profiles"
    latent_plot_dir.mkdir(parents=True, exist_ok=True)
    latent_plot_path = latent_plot_dir / f"layer{layer}_k{k_sparse}.png"
    latent_stats = plot_latent_profile(latents[bias_mask], latents[~bias_mask], latent_plot_path, f"Layer {layer} SAE Latent Means (k={k_sparse})")
    return {
        "plot": str(latent_plot_path), "biased_means": latent_stats["biased_means"].tolist(),
        "neutral_means": latent_stats["neutral_means"].tolist(), "mean_difference": latent_stats["mean_difference"].tolist(),
        "num_latents": latent_stats["num_latents"], "num_biased_samples": latent_stats["num_biased_samples"],
        "num_neutral_samples": latent_stats["num_neutral_samples"],
    }


def _process_k_sparse(k_sparse, activations, gld_scores, model, tokenizer, prompts, entries, bias_mask, activations_tensor,
                      layer, d_latent, device, suppression_scale, top_latents, text_corpus, mmlu_samples, baseline_mmlu_metrics, output_dir):
    """Process a single k_sparse value: train SAE, evaluate, and save results."""
    console.print(Panel(f"Training SAE with k_sparse={k_sparse}", style="bold blue"))
    sae, _ = train_sae(activations, model.cfg.d_model, d_latent or model.cfg.d_model * 8, k_sparse, device, epochs=1000)
    
    correlations = compute_latent_correlations(sae, activations, np.array(gld_scores))
    suppressed_latents = select_bias_latents(correlations, min_abs_corr=0.05, top_k=top_latents)
    if not suppressed_latents:
        console.print("[yellow]No latents exceeded correlation threshold; skipping suppression.[/yellow]")
        return None

    bias_metrics = evaluate_bias_reduction(model, tokenizer, prompts, sae, suppressed_latents, layer, suppression_scale)
    perplexity_metrics = evaluate_perplexity(model, tokenizer, text_corpus, sae, suppressed_latents, layer, suppression_scale)
    latent_profile_info = _compute_latent_profile(sae, activations_tensor, bias_mask, layer, k_sparse, output_dir)

    mmlu_result = None
    if mmlu_samples and baseline_mmlu_metrics:
        hook_name = "blocks.0.hook_resid_pre" if layer == 0 else f"blocks.{layer-1}.hook_resid_post"
        steered_metrics = evaluate_mmlu_accuracy(
            model, tokenizer, mmlu_samples,
            steering_hooks=[(hook_name, create_suppression_hook(sae, suppressed_latents, suppression_scale))]
        )
        mmlu_result = {"baseline": copy.deepcopy(baseline_mmlu_metrics), "steered": steered_metrics}
        console.print(f"[cyan]MMLU accuracy:[/cyan] {baseline_mmlu_metrics['overall_accuracy']:.3f} (baseline) → [green]{steered_metrics['overall_accuracy']:.3f}[/green] (steered)")

    result = {"k_sparse": k_sparse, "suppressed_latents": suppressed_latents, "bias_metrics": bias_metrics, "perplexity_metrics": perplexity_metrics}
    if mmlu_result:
        result["mmlu_accuracy"] = mmlu_result
    if latent_profile_info:
        result["latent_profile"] = latent_profile_info

    save_path = output_dir / f"sae_bias_suppression_k{k_sparse}.json"
    with open(save_path, "w") as f:
        json.dump(result, f, indent=2)
    console.print(f"[green]✓[/green] Saved result to {save_path}")
    return result


def run_experiment(
    model_name: str,
    dataset_name: str,
    layer: int,
    d_latent: Optional[int],
    k_sparse_values: List[int],
    n_examples: int,
    suppression_scale: float,
    top_latents: int,
    output_dir: Path,
    mmlu_subjects: Optional[Sequence[str]] = None,
    mmlu_max_questions: int = 25,
    skip_mmlu: bool = False,
    plot_activation_hist: bool = True,
) -> None:
    device = setup_device()
    model = load_model(model_name)
    model.to(device)
    tokenizer = get_tokenizer(model_name)

    output_dir.mkdir(parents=True, exist_ok=True)

    activations, gld_scores, entries = collect_activation_dataset(
        model=model,
        tokenizer=tokenizer,
        dataset_name=dataset_name,
        layer=layer,
        max_examples=n_examples,
    )
    prompts = [entry["prompt"] for entry in entries]

    activations_tensor = torch.from_numpy(activations).to(device)
    bias_mask = np.array([e.get("male_prob", 0.0) > e.get("female_prob", 0.0) for e in entries], dtype=bool)
    text_corpus = prompts[:min(50, len(prompts))]

    histogram_stats = _create_activation_histogram(model, tokenizer, entries, layer, output_dir) if plot_activation_hist else None
    mmlu_samples, baseline_mmlu_metrics, effective_subjects = (
        _load_mmlu_baseline(model, tokenizer, mmlu_subjects, mmlu_max_questions) if not skip_mmlu else ([], None, None)
    )

    results = [r for k in k_sparse_values if (r := _process_k_sparse(
        k, activations, gld_scores, model, tokenizer, prompts, entries, bias_mask, activations_tensor,
        layer, d_latent, device, suppression_scale, top_latents, text_corpus, mmlu_samples, baseline_mmlu_metrics, output_dir
    )) is not None]

    summary = {
        "model": model_name, "dataset": dataset_name, "layer": layer,
        "d_latent": d_latent or model.cfg.d_model * 8, "suppression_scale": suppression_scale, "top_latents": top_latents,
        "mmlu_subjects": effective_subjects if mmlu_samples else [], "baseline_mmlu_accuracy": baseline_mmlu_metrics,
        "activation_histogram": histogram_stats, "results": results,
    }
    summary_path = output_dir / "sae_bias_suppression_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    console.print(f"\n[bold green]✓[/bold green] Summary saved to {summary_path}")


@app.command()
def main(
    model: str = typer.Option("gpt2-medium", "--model", "-m", help="Model to use: gpt2-medium, gpt2-large, gpt-neo-125M, llama-3.2-1b"),
    dataset: str = typer.Option("stereoset_gender", "--dataset", "-d", help="Dataset name"),
    layer: int = typer.Option(0, "--layer", "-l", help="Layer index"),
    d_latent: Optional[int] = typer.Option(None, "--d-latent", help="Latent dimension"),
    k_sparse: List[int] = typer.Option([64], "--k-sparse", "-k", help="K-sparse values"),
    n_examples: int = typer.Option(2000, "--n-examples", "-n", help="Number of examples"),
    suppression_scale: float = typer.Option(0.0, "--suppression-scale", "-s", help="Suppression scale"),
    top_latents: int = typer.Option(32, "--top-latents", "-t", help="Number of top latents"),
    output_dir: Optional[str] = typer.Option(None, "--output-dir", "-o", help="Output directory"),
    mmlu_subjects: str = typer.Option(",".join(DEFAULT_MMLU_SUBJECTS), "--mmlu-subjects", help="MMLU subjects (comma-separated)"),
    mmlu_max_questions: int = typer.Option(25, "--mmlu-max-questions", help="Max MMLU questions per subject"),
    no_mmlu: bool = typer.Option(False, "--no-mmlu", help="Skip MMLU evaluation"),
    no_activation_hist: bool = typer.Option(False, "--no-activation-hist", help="Disable activation histogram"),
):
    """Run SAE steering experiment (Hegde 2024)."""
    valid_datasets = ["stereoset_gender", "winogender"]
    
    if dataset not in valid_datasets:
        console.print(f"[red]Error:[/red] Invalid dataset '{dataset}'. Must be one of {valid_datasets}")
        raise typer.Exit(1)
    
    console.print(Panel.fit("[bold cyan]SAE Steering Experiment[/bold cyan]", style="bold blue"))
    
    output_path = Path(output_dir) if output_dir else Path("results") / model / "sae_bias_suppression"
    subjects = [s.strip() for s in mmlu_subjects.split(",") if s.strip()] if mmlu_subjects else []
    
    table = Table(title="Experiment Configuration", show_header=True, header_style="bold magenta")
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Model", model)
    table.add_row("Dataset", dataset)
    table.add_row("Layer", str(layer))
    table.add_row("K-sparse", ", ".join(map(str, k_sparse)))
    table.add_row("Examples", str(n_examples))
    table.add_row("Output Dir", str(output_path))
    console.print(table)
    console.print()
    
    run_experiment(
        model_name=model,
        dataset_name=dataset,
        layer=layer,
        d_latent=d_latent,
        k_sparse_values=k_sparse,
        n_examples=n_examples,
        suppression_scale=suppression_scale,
        top_latents=top_latents,
        output_dir=output_path,
        mmlu_subjects=subjects,
        mmlu_max_questions=mmlu_max_questions,
        skip_mmlu=no_mmlu,
        plot_activation_hist=not no_activation_hist,
    )


if __name__ == "__main__":
    app()