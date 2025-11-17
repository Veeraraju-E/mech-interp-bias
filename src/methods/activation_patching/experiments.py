"""Main experiment orchestration for bias analysis."""

import json
import torch
from typing import Dict, List, Any
from pathlib import Path
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model_setup import load_model, get_tokenizer, setup_device
from src.data_loader import (
    load_stereoset,
    load_winogender,
    build_stereoset_triplets,
    build_winogender_pairs,
    find_first_difference_tokens
)
from src.bias_metrics import compute_bias_metric, build_bias_metric_fn
from src.methods.activation_patching import attribution_patch, scan_all_heads, scan_all_mlps


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
    elif dataset_name == "winogender":
        return _prepare_winogender_examples(examples, tokenizer)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

def run_attribution_patching_experiment(
    model,
    dataset_name: str,
    prepared_examples: List[Dict[str, Any]],
    bias_metric_fn,
    output_dir: Path
) -> Dict[str, Any]:
    """Run attribution patching using dataset-specific prepared inputs."""
    print(f"Running attribution patching experiment on {dataset_name} ({len(prepared_examples)} prompts)...")
    if not prepared_examples:
        print("Warning: no prepared examples found; skipping attribution patching.")
        return {"attributions": {}, "ranked_edges": []}
    
    attributions = attribution_patch(model, prepared_examples, bias_metric_fn)
    
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
    output_dir: Path
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
    
    print("Scanning all attention heads...")
    head_impacts = scan_all_heads(model, prepared_examples, bias_metric_fn)
    
    print("Scanning all MLPs...")
    mlp_impacts = scan_all_mlps(model, prepared_examples, bias_metric_fn)
    
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
    print("Initializing model and datasets...")
    
    device = setup_device()
    model = load_model()
    model.to(device)
    tokenizer = get_tokenizer()
    
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    
    print("Loading datasets...")
    stereoset_examples = load_stereoset()
    winogender_examples = load_winogender()
    
    print(f"Loaded {len(stereoset_examples)} StereoSet examples")
    print(f"Loaded {len(winogender_examples)} WinoGender examples")
    
    print("\nComputing baseline bias metrics...")
    stereoset_baseline = compute_bias_metric(model, stereoset_examples, "stereoset", tokenizer)
    winogender_baseline = compute_bias_metric(model, winogender_examples, "winogender", tokenizer)
    
    print(f"StereoSet baseline bias: {stereoset_baseline:.4f}")
    print(f"WinoGender baseline bias: {winogender_baseline:.4f}")
    
    datasets = [
        ("stereoset", stereoset_examples),
        ("winogender", winogender_examples)
    ]
    
    all_results = {}
    
    for dataset_name, examples in datasets:
        print(f"\n{'='*60}")
        print(f"Running experiments on {dataset_name}")
        print(f"{'='*60}")
        
        prepared_examples = prepare_dataset_inputs(dataset_name, examples, tokenizer)
        bias_metric_fn = build_bias_metric_fn(dataset_name)
        
        attribution_results = run_attribution_patching_experiment(
            model, dataset_name, prepared_examples, bias_metric_fn, output_dir
        )
        
        ablation_results = run_ablation_experiment(
            model, dataset_name, prepared_examples, bias_metric_fn, output_dir
        )
        
        all_results[dataset_name] = {
            "baseline_bias": stereoset_baseline if dataset_name == "stereoset" else winogender_baseline,
            "attribution_patching": attribution_results,
            "ablations": ablation_results
        }
    
    summary_file = output_dir / "summary.json"
    with open(summary_file, "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("Experiments complete! Results saved to", output_dir)
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

