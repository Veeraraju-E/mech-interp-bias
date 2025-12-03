"""Main experiment for Activation Patching."""

import json
from typing import Dict, List, Any, Optional
from pathlib import Path
import sys
import os
import torch
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.bias_metrics import compute_bias_metric, build_bias_metric_fn
from src.model_setup import *
from src.data_loader import *
from src.methods.patching import *
from src.cache_utils import *

app = typer.Typer(help="Activation patching experiments for bias analysis")
console = Console()

def _prepare_stereoset_examples(
    examples: List[Dict[str, Any]],
    tokenizer
) -> List[Dict[str, Any]]:
    """Tokenize StereoSet contexts and attach per-example target token metadata."""
    triplets = build_stereoset_triplets(examples)
    prepared = []
    for triplet in triplets:
        prompt = triplet["context_prefix"] or triplet["context"]
        if not prompt:
            continue
        tokens = tokenizer.encode(prompt, return_tensors="pt").squeeze(0)
        if tokens.numel() == 0:
            continue
        
        stereo_word, anti_word = find_first_difference_tokens(
            triplet["stereotype_sentence"],
            triplet["antistereotype_sentence"]
        )
        if not stereo_word or not anti_word:
            continue
        
        stereo_token_ids = tokenizer.encode(stereo_word, add_special_tokens=False)
        antistereo_token_ids = tokenizer.encode(anti_word, add_special_tokens=False)
        if not stereo_token_ids or not antistereo_token_ids:
            continue
        
        metadata = {
            "bias_type": triplet["bias_type"],
            "target": triplet["target"],
            "stereo_token_ids": stereo_token_ids,
            "antistereo_token_ids": antistereo_token_ids,
            "stereotype_word": stereo_word,
            "antistereotype_word": anti_word
        }
        prepared.append({"tokens": tokens, "metadata": metadata})
    return prepared


def _prepare_winogender_examples(
    examples: List[Dict[str, Any]],
    tokenizer
) -> List[Dict[str, Any]]:
    """Tokenize WinoGender contexts and store pronoun token ids."""
    pairs = build_winogender_pairs(examples)
    prepared = []
    for pair in pairs:
        prompt = pair["prompt"]
        if not prompt:
            continue
        tokens = tokenizer.encode(prompt, return_tensors="pt").squeeze(0)
        if tokens.numel() == 0:
            continue
        
        male_token_ids = tokenizer.encode(pair["male_pronoun"], add_special_tokens=False)
        female_token_ids = tokenizer.encode(pair["female_pronoun"], add_special_tokens=False)
        if not male_token_ids or not female_token_ids:
            continue
        
        metadata = {
            "example_id": pair["example_id"],
            "profession": pair["profession"],
            "word": pair["word"],
            "male_token_ids": male_token_ids,
            "female_token_ids": female_token_ids
        }
        prepared.append({"tokens": tokens, "metadata": metadata})
    return prepared


def prepare_dataset_inputs(
    dataset_name: str,
    examples: List[Dict[str, Any]],
    tokenizer
) -> List[Dict[str, Any]]:
    """Dispatch to dataset-specific tokenization helpers."""
    if dataset_name == "stereoset":
        return _prepare_stereoset_examples(examples, tokenizer)
    elif dataset_name in ["stereoset_race", "stereoset_gender"]:
        acdc_pairs = get_acdc_stereoset_pairs("race" if dataset_name == "stereoset_race" else "gender")
        prepared = []
        for pair in acdc_pairs:
            clean_entry = pair.get("clean", {})
            corrupted_entry = pair.get("corrupted", {})
            
            if clean_entry and clean_entry.get("tokens"):
                clean_tokens = torch.tensor(clean_entry["tokens"], dtype=torch.long)
                clean_metadata = clean_entry.get("metadata", {})
                prepared.append({
                    "tokens": clean_tokens,
                    "metadata": clean_metadata
                })
            
            if corrupted_entry and corrupted_entry.get("tokens"):
                corrupted_tokens = torch.tensor(corrupted_entry["tokens"], dtype=torch.long)
                corrupted_metadata = corrupted_entry.get("metadata", {})
                prepared.append({
                    "tokens": corrupted_tokens,
                    "metadata": corrupted_metadata
                })
        
        return prepared
    elif dataset_name == "winogender":
        return _prepare_winogender_examples(examples, tokenizer)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

def run_attribution_patching_experiment(
    model,
    dataset_name: str,
    prepared_examples: List[Dict[str, Any]],
    bias_metric_fn,
    output_dir: Path,
    cache_dir: Optional[Path] = None,
    use_cache: bool = True
) -> Dict[str, Any]:
    """Run attribution patching using dataset-specific prepared inputs."""
    console.print(f"[cyan]Running attribution patching[/cyan] on [bold]{dataset_name}[/bold] ([dim]{len(prepared_examples)} prompts[/dim])...")
    if not prepared_examples:
        console.print("[yellow]Warning:[/yellow] no prepared examples found; skipping attribution patching.")
        return {"attributions": {}, "ranked_edges": []}
    
    model_name = get_model_name(model)
    attributions = None
    
    if cache_dir and use_cache:
        cached_attributions = load_cached_attribution_results(cache_dir, model_name, dataset_name)
        if cached_attributions is not None:
            attributions = cached_attributions
            console.print("[green]✓[/green] Using cached attribution results")
    
    if attributions is None:
        with console.status("[bold yellow]Computing attributions..."):
            attributions = attribution_patch(model, prepared_examples, bias_metric_fn)
        
        if cache_dir and use_cache:
            cache_attribution_results(cache_dir, model_name, dataset_name, attributions, use_cache)
    
    results_file = output_dir / f"attribution_patching_{dataset_name}.json"
    with open(results_file, "w") as f:
        json.dump(attributions, f, indent=2)
    
    ranked_edges = sorted(attributions.items(), key=lambda x: abs(x[1]), reverse=True)
    
    table = Table(title="Top 10 Attribution Edges", show_header=True, header_style="bold magenta")
    table.add_column("Hook Name", style="cyan")
    table.add_column("Score", style="green", justify="right")
    for hook_name, score in ranked_edges[:10]:
        table.add_row(hook_name, f"{score:.4f}")
    console.print(table)
    
    return {
        "attributions": attributions,
        "ranked_edges": ranked_edges[:20]
    }


def run_ablation_experiment(
    model,
    dataset_name: str,
    prepared_examples: List[Dict[str, Any]],
    bias_metric_fn,
    output_dir: Path,
    cache_dir: Optional[Path] = None,
    use_cache: bool = True
) -> Dict[str, Any]:
    """Run head/MLP ablations using the same prepared prompts as attribution runs."""
    console.print(f"[cyan]Running ablation experiment[/cyan] on [bold]{dataset_name}[/bold]...")
    if not prepared_examples:
        console.print("[yellow]Warning:[/yellow] no prepared examples found; skipping ablations.")
        return {
            "head_impacts": {},
            "mlp_impacts": {},
            "ranked_heads": [],
            "ranked_mlps": []
        }
    
    model_name = get_model_name(model)
    head_impacts = None
    mlp_impacts = None
    
    if cache_dir and use_cache:
        cached_results = load_cached_ablation_results(cache_dir, model_name, dataset_name)
        if cached_results is not None:
            cached_heads, cached_mlps = cached_results
            head_impacts = {}
            for k, v in cached_heads.items():
                if isinstance(k, str) and '_' in k:
                    parts = k.split('_')
                    if len(parts) == 2:
                        head_impacts[(int(parts[0]), int(parts[1]))] = v
                    else:
                        head_impacts[k] = v
                else:
                    head_impacts[k] = v
            mlp_impacts = {int(k) if isinstance(k, str) else k: v for k, v in cached_mlps.items()}
            console.print("[green]✓[/green] Using cached ablation results")
    
    if head_impacts is None or mlp_impacts is None:
        if head_impacts is None:
            with console.status("[bold yellow]Scanning all attention heads..."):
                head_impacts = scan_all_heads(model, prepared_examples, bias_metric_fn)
        
        if mlp_impacts is None:
            with console.status("[bold yellow]Scanning all MLPs..."):
                mlp_impacts = scan_all_mlps(model, prepared_examples, bias_metric_fn)
        
        if cache_dir and use_cache:
            head_dict = {f"{layer}_{head}": score for (layer, head), score in head_impacts.items()}
            cache_ablation_results(cache_dir, model_name, dataset_name, head_dict, mlp_impacts, use_cache)
    
    head_file = output_dir / f"head_ablations_{dataset_name}.json"
    with open(head_file, "w") as f:
        head_dict = {f"{layer}_{head}": score for (layer, head), score in head_impacts.items()}
        json.dump(head_dict, f, indent=2)
    
    mlp_file = output_dir / f"mlp_ablations_{dataset_name}.json"
    with open(mlp_file, "w") as f:
        json.dump(mlp_impacts, f, indent=2)
    
    ranked_heads = sorted(head_impacts.items(), key=lambda x: abs(x[1]), reverse=True)
    ranked_mlps = sorted(mlp_impacts.items(), key=lambda x: abs(x[1]), reverse=True)
    
    head_table = Table(title="Top 10 Biased Heads", show_header=True, header_style="bold magenta")
    head_table.add_column("Layer", style="cyan")
    head_table.add_column("Head", style="cyan")
    head_table.add_column("Impact", style="green", justify="right")
    for (layer, head), impact in ranked_heads[:10]:
        head_table.add_row(str(layer), str(head), f"{impact:.4f}")
    console.print(head_table)
    
    mlp_table = Table(title="Top 10 Biased MLPs", show_header=True, header_style="bold magenta")
    mlp_table.add_column("Layer", style="cyan")
    mlp_table.add_column("Impact", style="green", justify="right")
    for layer, impact in ranked_mlps[:10]:
        mlp_table.add_row(str(layer), f"{impact:.4f}")
    console.print(mlp_table)
    
    head_impacts_json = {f"{layer}_{head}": score for (layer, head), score in head_impacts.items()}
    
    return {
        "head_impacts": head_impacts_json,
        "mlp_impacts": mlp_impacts,
        "ranked_heads": [(f"{layer}_{head}", impact) for (layer, head), impact in ranked_heads],
        "ranked_mlps": ranked_mlps
    }


@app.command()
def main(
    model: str = typer.Option("gpt2-medium", "--model", "-m", help="Model to use: gpt2-medium, gpt2-large, gpt-neo-125M, llama-3.2-1b"),
    no_cache: bool = typer.Option(False, "--no-cache", help="Disable caching and recompute everything"),
    cache_dir: str = typer.Option("runs/activation_patching/cache", "--cache-dir", help="Directory to store/load cached components"),
    output_dir: str = typer.Option("results", "--output-dir", "-o", help="Directory to save results: results/<model>/activation_patching"),
    datasets: List[str] = typer.Option(["stereoset_race", "stereoset_gender", "winogender"], "--datasets", "-d", help="List of datasets to run experiments on. Options: stereoset_race, stereoset_gender, winogender"),
):
    """Main experiment orchestration for activation patching."""
    console.print(Panel.fit("[bold cyan]Activation Patching Experiments[/bold cyan]", style="bold blue"))
    
    # Validate datasets
    valid_datasets = {"stereoset_race", "stereoset_gender", "winogender"}
    datasets_set = set(datasets)
    invalid_datasets = datasets_set - valid_datasets
    if invalid_datasets:
        console.print(f"[red]Error:[/red] Invalid dataset(s): {', '.join(invalid_datasets)}")
        console.print(f"[yellow]Valid datasets:[/yellow] {', '.join(sorted(valid_datasets))}")
        raise typer.Exit(1)
    
    if not datasets_set:
        console.print("[red]Error:[/red] At least one dataset must be specified")
        raise typer.Exit(1)
    
    device = setup_device()
    with console.status("[bold yellow]Loading model..."):
        model_obj = load_model(model)
        model_obj.to(device)
    tokenizer = get_tokenizer(model)
    model_name = get_model_name(model_obj)
    
    output_path = Path(output_dir) / model_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    cache_path = Path(cache_dir) if not no_cache else None
    use_cache = not no_cache
    
    table = Table(title="Configuration", show_header=True, header_style="bold magenta")
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Model", model)
    table.add_row("Datasets", ", ".join(sorted(datasets_set)))
    table.add_row("Output Dir", str(output_path))
    table.add_row("Cache", "Enabled" if use_cache else "Disabled")
    if cache_path:
        table.add_row("Cache Dir", str(cache_path))
    console.print(table)
    console.print()
    
    with console.status("[bold yellow]Loading datasets..."):
        stereoset_examples = []
        if "stereoset_race" in datasets_set or "stereoset_gender" in datasets_set:
            stereoset_examples = load_stereoset()
            console.print(f"[green]✓[/green] Loaded {len(stereoset_examples)} StereoSet examples")
        
        winogender_examples = []
        if "winogender" in datasets_set:
            winogender_examples = load_winogender()
            console.print(f"[green]✓[/green] Loaded {len(winogender_examples)} WinoGender examples")
    console.print()
    
    console.print("[cyan]Computing baseline bias metrics...[/cyan]")
    baseline_scores = {}
    
    # Only compute baselines for selected datasets
    for bias_type in ["race", "gender"]:
        dataset_name = f"stereoset_{bias_type}"
        if dataset_name not in datasets_set:
            continue
            
        filtered_examples = [ex for ex in stereoset_examples if ex.get("bias_type", "").lower() == bias_type]
        
        if cache_path and use_cache:
            cached_scores = load_cached_bias_scores(cache_path, model_name, dataset_name)
            if cached_scores is not None:
                baseline_scores[dataset_name] = cached_scores.get("baseline", None)
                console.print(f"[green]✓[/green] Loaded cached baseline for {dataset_name}: {baseline_scores[dataset_name]:.4f}")
        
        if dataset_name not in baseline_scores or baseline_scores[dataset_name] is None:
            if filtered_examples:
                baseline = compute_bias_metric(model_obj, filtered_examples, "stereoset", tokenizer)
                baseline_scores[dataset_name] = baseline
                console.print(f"[cyan]{dataset_name.capitalize()}[/cyan] baseline bias: [bold]{baseline:.4f}[/bold]")
                
                if cache_path and use_cache:
                    cache_bias_scores(cache_path, model_name, dataset_name, {"baseline": baseline}, use_cache)
            else:
                baseline_scores[dataset_name] = 0.0
                console.print(f"[yellow]Warning:[/yellow] No examples found for {dataset_name}, setting baseline to 0.0")
    
    if "winogender" in datasets_set:
        dataset_name = "winogender"
        if cache_path and use_cache:
            cached_scores = load_cached_bias_scores(cache_path, model_name, dataset_name)
            if cached_scores is not None:
                baseline_scores[dataset_name] = cached_scores.get("baseline", None)
                console.print(f"[green]✓[/green] Loaded cached baseline for {dataset_name}: {baseline_scores[dataset_name]:.4f}")
        
        if dataset_name not in baseline_scores or baseline_scores[dataset_name] is None:
            baseline = compute_bias_metric(model_obj, winogender_examples, dataset_name, tokenizer)
            baseline_scores[dataset_name] = baseline
            console.print(f"[cyan]{dataset_name.capitalize()}[/cyan] baseline bias: [bold]{baseline:.4f}[/bold]")
            
            if cache_path and use_cache:
                cache_bias_scores(cache_path, model_name, dataset_name, {"baseline": baseline}, use_cache)
    
    # Build dataset list based on selected datasets
    datasets = []
    if "stereoset_race" in datasets_set:
        datasets.append(("stereoset_race", None))
    if "stereoset_gender" in datasets_set:
        datasets.append(("stereoset_gender", None))
    if "winogender" in datasets_set:
        datasets.append(("winogender", winogender_examples))
    
    all_results = {}
    
    for dataset_name, examples in datasets:
        console.print(Panel(f"[bold cyan]Running experiments on {dataset_name}[/bold cyan]", style="bold blue"))
        
        prepared_examples = None
        
        if cache_path and use_cache:
            cached_examples = load_cached_prepared_examples(cache_path, model_name, dataset_name)
            if cached_examples is not None:
                prepared_examples = cached_examples
                console.print(f"[green]✓[/green] Loaded {len(prepared_examples)} cached prepared examples")
        
        if prepared_examples is None:
            if dataset_name in ["stereoset_race", "stereoset_gender"]:
                prepared_examples = prepare_dataset_inputs(dataset_name, None, tokenizer)
            else:
                prepared_examples = prepare_dataset_inputs(dataset_name, examples, tokenizer)
            if cache_path and use_cache:
                cache_prepared_examples(cache_path, model_name, dataset_name, prepared_examples, use_cache)
        
        bias_metric_fn = build_bias_metric_fn("stereoset" if "stereoset" in dataset_name else dataset_name)
        
        attribution_results = run_attribution_patching_experiment(model_obj, dataset_name, prepared_examples, bias_metric_fn, output_path, cache_dir=cache_path, use_cache=use_cache)
        
        ablation_results = run_ablation_experiment(model_obj, dataset_name, prepared_examples, bias_metric_fn, output_path, cache_dir=cache_path, use_cache=use_cache)
        
        baseline = baseline_scores.get(dataset_name, 0.0)
        all_results[dataset_name] = {
            "baseline_bias": baseline,
            "attribution_patching": attribution_results,
            "ablations": ablation_results
        }
        console.print()
    
    summary_file = output_path / "summary.json"
    with open(summary_file, "w") as f:
        json.dump(all_results, f, indent=2)
    
    console.print(Panel.fit(f"[bold green]✓ Experiments complete![/bold green]\nResults saved to [cyan]{output_path}[/cyan]", style="bold green"))
    if cache_path and use_cache:
        console.print(f"[dim]Cache saved to {cache_path}[/dim]")


if __name__ == "__main__":
    app()