"""Main experiment for Activation Patching."""

import json
import argparse
from typing import Dict, List, Any, Optional
from pathlib import Path
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.bias_metrics import compute_bias_metric, build_bias_metric_fn
from src.model_setup import *
from src.data_loader import *
from src.methods.activation_patching import *
from src.cache_utils import *

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
    elif dataset_name == "stereoset_race":
        return load_stereoset_acdc_pairs("race")
    elif dataset_name == "stereoset_gender":
        return load_stereoset_acdc_pairs("gender")
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
    print(f"Running attribution patching experiment on {dataset_name} ({len(prepared_examples)} prompts)...")
    if not prepared_examples:
        print("Warning: no prepared examples found; skipping attribution patching.")
        return {"attributions": {}, "ranked_edges": []}
    
    model_name = get_model_name(model)
    attributions = None
    
    if cache_dir and use_cache:
        cached_attributions = load_cached_attribution_results(cache_dir, model_name, dataset_name)
        if cached_attributions is not None:
            attributions = cached_attributions
            print("Using cached attribution results")
    
    if attributions is None:
        attributions = attribution_patch(model, prepared_examples, bias_metric_fn)
        
        if cache_dir and use_cache:
            cache_attribution_results(cache_dir, model_name, dataset_name, attributions, use_cache)
    
    results_file = output_dir / f"attribution_patching_{dataset_name}.json"
    with open(results_file, "w") as f:
        json.dump(attributions, f, indent=2)
    
    ranked_edges = sorted(attributions.items(), key=lambda x: abs(x[1]), reverse=True)
    
    print(f"Top 10 attribution edges:")
    for hook_name, score in ranked_edges[:10]:
        print(f"  {hook_name}: {score:.4f}")
    
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
    print(f"Running ablation experiment on {dataset_name}...")
    if not prepared_examples:
        print("Warning: no prepared examples found; skipping ablations.")
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
            print("Using cached ablation results")
    
    if head_impacts is None or mlp_impacts is None:
        if head_impacts is None:
            print("Scanning all attention heads...")
            head_impacts = scan_all_heads(model, prepared_examples, bias_metric_fn)
        
        if mlp_impacts is None:
            print("Scanning all MLPs...")
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
    
    print(f"Top 10 biased heads:")
    for (layer, head), impact in ranked_heads[:10]:
        print(f"  Layer {layer}, Head {head}: {impact:.4f}")
    
    print(f"Top 10 biased MLPs:")
    for layer, impact in ranked_mlps[:10]:
        print(f"  Layer {layer}: {impact:.4f}")
    
    head_impacts_json = {f"{layer}_{head}": score for (layer, head), score in head_impacts.items()}
    
    return {
        "head_impacts": head_impacts_json,
        "mlp_impacts": mlp_impacts,
        "ranked_heads": [(f"{layer}_{head}", impact) for (layer, head), impact in ranked_heads],
        "ranked_mlps": ranked_mlps
    }


def main():
    """Main experiment orchestration."""
    parser = argparse.ArgumentParser(description="Activation patching experiments for bias analysis")
    parser.add_argument(
        "--model",
        type=str,
        default="gpt2-medium",
        choices=["gpt2-medium", "gpt2-large"],
        help="Model to use (default: gpt2-medium)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt2-medium",
        choices=["gpt2-medium", "gpt2-large"],
        help="Model to use (default: gpt2-medium)"
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable caching and recompute everything from scratch"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="runs/activation_patching/cache",
        help="Directory to store/load cached components (default: runs/activation_patching/cache)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory to save results (default: results)"
    )
    
    args = parser.parse_args()
    
    print("Initializing model and datasets...")
    print(f"Using model: {args.model}")
    print(f"Using model: {args.model}")
    
    device = setup_device()
    model = load_model(args.model)
    model = load_model(args.model)
    model.to(device)
    tokenizer = get_tokenizer(args.model)
    
    model_name = get_model_name(model)
    tokenizer = get_tokenizer(args.model)
    model_name = get_model_name(model)
    
    output_dir = Path(args.output_dir) / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_dir = Path(args.output_dir) / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cache_dir = Path(args.cache_dir) if not args.no_cache else None
    use_cache = not args.no_cache
    
    if cache_dir:
        print(f"Cache directory: {cache_dir}")
        if use_cache:
            print("Caching enabled - will load from cache if available")
        else:
            print("Caching disabled - will recompute everything")
    
    print("Loading datasets...")
    stereoset_examples = load_stereoset()
    winogender_examples = load_winogender()
    
    print(f"Loaded {len(stereoset_examples)} StereoSet examples")
    print(f"Loaded {len(winogender_examples)} WinoGender examples")
    
    print("\nComputing baseline bias metrics...")
    baseline_scores = {}
    
    for bias_type in ["race", "gender"]:
        dataset_name = f"stereoset_{bias_type}"
        filtered_examples = [ex for ex in stereoset_examples if ex.get("bias_type", "").lower() == bias_type]
        
        if cache_dir and use_cache:
            cached_scores = load_cached_bias_scores(cache_dir, model_name, dataset_name)
            if cached_scores is not None:
                baseline_scores[dataset_name] = cached_scores.get("baseline", None)
                print(f"Loaded cached baseline for {dataset_name}: {baseline_scores[dataset_name]:.4f}")
        
        if dataset_name not in baseline_scores or baseline_scores[dataset_name] is None:
            if filtered_examples:
                baseline = compute_bias_metric(model, filtered_examples, "stereoset", tokenizer)
                baseline_scores[dataset_name] = baseline
                print(f"{dataset_name.capitalize()} baseline bias: {baseline:.4f}")
                
                if cache_dir and use_cache:
                    cache_bias_scores(cache_dir, model_name, dataset_name, {"baseline": baseline}, use_cache)
            else:
                baseline_scores[dataset_name] = 0.0
                print(f"Warning: No examples found for {dataset_name}, setting baseline to 0.0")
    
    dataset_name = "winogender"
    if cache_dir and use_cache:
        cached_scores = load_cached_bias_scores(cache_dir, model_name, dataset_name)
        if cached_scores is not None:
            baseline_scores[dataset_name] = cached_scores.get("baseline", None)
            print(f"Loaded cached baseline for {dataset_name}: {baseline_scores[dataset_name]:.4f}")
    
    if dataset_name not in baseline_scores or baseline_scores[dataset_name] is None:
        baseline = compute_bias_metric(model, winogender_examples, dataset_name, tokenizer)
        baseline_scores[dataset_name] = baseline
        print(f"{dataset_name.capitalize()} baseline bias: {baseline:.4f}")
        
        if cache_dir and use_cache:
            cache_bias_scores(cache_dir, model_name, dataset_name, {"baseline": baseline}, use_cache)
    
    datasets = [("stereoset_race", None), ("stereoset_gender", None), ("winogender", winogender_examples)]
    
    all_results = {}
    
    for dataset_name, examples in datasets:
        print(f"\n{'='*60}")
        print(f"Running experiments on {dataset_name}")
        print(f"{'='*60}")
        
        prepared_examples = None
        
        if cache_dir and use_cache:
            cached_examples = load_cached_prepared_examples(cache_dir, model_name, dataset_name)
            if cached_examples is not None:
                prepared_examples = cached_examples
                print(f"Loaded {len(prepared_examples)} cached prepared examples")
        
        if prepared_examples is None:
            if dataset_name in ["stereoset_race", "stereoset_gender"]:
                prepared_examples = prepare_dataset_inputs(dataset_name, None, tokenizer)
            else:
                prepared_examples = prepare_dataset_inputs(dataset_name, examples, tokenizer)
            if cache_dir and use_cache:
                cache_prepared_examples(cache_dir, model_name, dataset_name, prepared_examples, use_cache)
        
        bias_metric_fn = build_bias_metric_fn("stereoset" if "stereoset" in dataset_name else dataset_name)
        
        attribution_results = run_attribution_patching_experiment(
            model, dataset_name, prepared_examples, bias_metric_fn, output_dir,
            cache_dir=cache_dir, use_cache=use_cache
        )
        
        ablation_results = run_ablation_experiment(
            model, dataset_name, prepared_examples, bias_metric_fn, output_dir,
            cache_dir=cache_dir, use_cache=use_cache
        )
        
        baseline = baseline_scores.get(dataset_name, 0.0)
        all_results[dataset_name] = {
            "baseline_bias": baseline,
            "attribution_patching": attribution_results,
            "ablations": ablation_results
        }
    
    summary_file = output_dir / "summary.json"
    with open(summary_file, "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("Experiments complete! Results saved to", output_dir)
    if cache_dir and use_cache:
        print(f"Cache saved to {cache_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

