"""Main experiment orchestration for bias activation patching."""

import json
import csv
import torch
from typing import Dict, List, Any
from pathlib import Path
import torch.nn.functional as F

from .model_setup import load_model, get_tokenizer, setup_device
from .data_loader import load_stereoset, load_winogender, prepare_prompts, get_stereoset_pairs
from .bias_metrics import compute_bias_metric, compute_bias_metric_for_patching
from .activation_patching import scan_all_edges, get_all_hook_points
from .attribution_patching import attribution_patch, validate_top_edges
from .ablations import scan_all_heads, scan_all_mlps


def create_bias_metric_fn(model, tokenizer, dataset_name: str, target_tokens: Dict[str, List[int]] = None):
    """
    Create a bias metric function for use in patching experiments.
    
    Args:
        model: Model instance
        tokenizer: Tokenizer instance
        dataset_name: Name of dataset
        target_tokens: Optional dict with 'stereotype' and 'antistereotype' token IDs
    
    Returns:
        Function that computes bias from logits
    """
    def bias_metric_fn(logits: torch.Tensor) -> torch.Tensor:
        """
        Compute bias metric from logits.
        Returns a tensor that can be used for gradient computation.
        """
        if len(logits.shape) == 3:
            next_token_logits = logits[0, -1, :]
        else:
            next_token_logits = logits[-1, :]
        
        probs = F.softmax(next_token_logits, dim=-1)
        
        if dataset_name == "stereoset":
            # Compare log probabilities of stereotype vs antistereotype tokens
            if target_tokens and "stereotype" in target_tokens and "antistereotype" in target_tokens:
                stereo_token_ids = target_tokens["stereotype"]
                antistereo_token_ids = target_tokens["antistereotype"]
                
                # Sum probabilities for all stereotype tokens
                stereo_prob = sum(probs[tid].item() for tid in stereo_token_ids if tid < len(probs))
                # Sum probabilities for all antistereotype tokens
                antistereo_prob = sum(probs[tid].item() for tid in antistereo_token_ids if tid < len(probs))
                
                # Return log probability difference (positive = biased toward stereotype)
                stereo_log_prob = torch.log(torch.tensor(stereo_prob + 1e-10, device=logits.device))
                antistereo_log_prob = torch.log(torch.tensor(antistereo_prob + 1e-10, device=logits.device))
                return stereo_log_prob - antistereo_log_prob
            else:
                # Fallback: use entropy as proxy
                entropy = -(probs * torch.log(probs + 1e-10)).sum()
                return -entropy
        
        elif dataset_name == "winogender":
            # Measure pronoun prediction difference
            if len(logits.shape) == 3:
                next_token_logits = logits[0, -1, :]
            else:
                next_token_logits = logits[-1, :]
            
            probs = F.softmax(next_token_logits, dim=-1)
            
            # Get probabilities of gender pronouns
            male_pronouns = ["he", "him", "his"]
            female_pronouns = ["she", "her", "hers"]
            
            male_prob = torch.tensor(0.0, device=logits.device)
            female_prob = torch.tensor(0.0, device=logits.device)
            
            for p in male_pronouns:
                p_tokens = tokenizer.encode(p, add_special_tokens=False)
                if len(p_tokens) > 0:
                    male_prob = male_prob + probs[p_tokens[0]]
            
            for p in female_pronouns:
                p_tokens = tokenizer.encode(p, add_special_tokens=False)
                if len(p_tokens) > 0:
                    female_prob = female_prob + probs[p_tokens[0]]
            
            # Return difference (positive = male bias)
            return male_prob - female_prob
        
        else:
            # Fallback: use mean logit
            return logits.mean()
    
    return bias_metric_fn


def run_causal_patching_experiment(
    model,
    examples: List[Dict[str, Any]],
    dataset_name: str,
    tokenizer,
    output_dir: Path
) -> Dict[str, Any]:
    """
    Run causal activation patching experiment.
    
    Args:
        model: HookedTransformer model
        examples: List of dataset examples
        dataset_name: Name of dataset ("stereoset" or "winogender")
        tokenizer: Tokenizer instance
        output_dir: Directory to save results
    
    Returns:
        Dictionary of results
    """
    print(f"Running causal patching experiment on {dataset_name}...")
    
    # Prepare examples
    if dataset_name == "stereoset":
        pairs = get_stereoset_pairs(examples)
        if len(pairs) == 0:
            return {}
        
        # Create biased and neutral prompts
        # For StereoSet, we use the context as prompt and extract target words
        biased_prompts = []
        neutral_prompts = []
        target_words_list = []
        
        for pair in pairs:
            context = pair["context"]
            stereo_sent = pair["stereotype_sentence"]
            antistereo_sent = pair["antistereotype_sentence"]
            
            # Extract the target word (the word that differs between sentences)
            # Find the first word that's different after the context
            context_words = context.replace("BLANK", "").strip().split()
            stereo_words = stereo_sent.split()
            antistereo_words = antistereo_sent.split()
            
            # Find the differing word - sentences repeat context, so find first different word
            stereo_word = None
            antistereo_word = None
            
            # Remove context prefix from sentences to find target words
            context_prefix = context.replace("BLANK", "").strip()
            stereo_clean = stereo_sent.replace(context_prefix, "").strip()
            antistereo_clean = antistereo_sent.replace(context_prefix, "").strip()
            
            stereo_words_clean = stereo_clean.split()
            antistereo_words_clean = antistereo_clean.split()
            
            for sw, aw in zip(stereo_words_clean, antistereo_words_clean):
                if sw != aw:
                    stereo_word = sw.rstrip('.,!?;:')
                    antistereo_word = aw.rstrip('.,!?;:')
                    break
            
            if stereo_word and antistereo_word:
                # Tokenize target words
                stereo_tokens = tokenizer.encode(stereo_word, add_special_tokens=False)
                antistereo_tokens = tokenizer.encode(antistereo_word, add_special_tokens=False)
                
                if stereo_tokens and antistereo_tokens:
                    target_words_list.append({
                        "stereotype": stereo_tokens,
                        "antistereotype": antistereo_tokens
                    })
                    
                    # For patching: use full sentences to get different activations
                    # Biased: context + stereotype sentence (to get biased activations)
                    # Neutral: context + antistereotype sentence (to measure bias)
                    biased_prompts.append(context + " " + stereo_sent)
                    neutral_prompts.append(context + " " + antistereo_sent)
                else:
                    target_words_list.append(None)
                    biased_prompts.append(context)
                    neutral_prompts.append(context)
            else:
                # Fallback: use context only
                target_words_list.append(None)
                biased_prompts.append(context)
                neutral_prompts.append(context)
    else:
        # For WinoGender, we need to create biased/neutral pairs
        # Group by profession and create male/female pronoun pairs
        profession_groups = {}
        for ex in examples[:50]:  # Use more examples
            prof = ex.get("profession", "")
            if not prof:
                continue
            if prof not in profession_groups:
                profession_groups[prof] = {"male": [], "female": []}
            
            answer = ex.get("answer", "").lower()
            if answer == "male" or ex.get("pronoun", "").lower() in ["he", "him", "his"]:
                profession_groups[prof]["male"].append(ex)
            elif answer == "female" or ex.get("pronoun", "").lower() in ["she", "her", "hers"]:
                profession_groups[prof]["female"].append(ex)
        
        # Create pairs: one with male pronoun, one with female pronoun for same profession
        biased_prompts = []
        neutral_prompts = []
        target_words_list = []
        
        for prof, groups in profession_groups.items():
            if groups["male"] and groups["female"]:
                # Use first male and first female example
                male_ex = groups["male"][0]
                female_ex = groups["female"][0]
                
                # Create prompts by replacing pronoun
                male_sent = male_ex["sentence"]
                female_sent = female_ex["sentence"]
                
                biased_prompts.append(male_sent)
                neutral_prompts.append(female_sent)
                
                # Extract pronouns as target tokens
                male_pronoun = male_ex.get("pronoun", "he")
                female_pronoun = female_ex.get("pronoun", "she")
                
                male_tokens = tokenizer.encode(male_pronoun, add_special_tokens=False)
                female_tokens = tokenizer.encode(female_pronoun, add_special_tokens=False)
                
                if male_tokens and female_tokens:
                    target_words_list.append({
                        "stereotype": male_tokens,  # Using male as "stereotype" for WinoGender
                        "antistereotype": female_tokens
                    })
                else:
                    target_words_list.append(None)
                
                if len(biased_prompts) >= 10:
                    break
        
        # If no pairs found, use sentences as-is
        if not biased_prompts:
            biased_prompts = [ex["sentence"] for ex in examples[:10]]
            neutral_prompts = [ex["sentence"] for ex in examples[:10]]
            target_words_list = []
    
    # Tokenize
    biased_tokens = [tokenizer.encode(p, return_tensors="pt").squeeze(0) for p in biased_prompts]
    neutral_tokens = [tokenizer.encode(p, return_tensors="pt").squeeze(0) for p in neutral_prompts]
    
    # Extract context length for proper patching
    context_len = None
    if dataset_name == "stereoset" and len(biased_prompts) > 0:
        # Get context from first pair
        if len(pairs) > 0:
            context = pairs[0]["context"]
            context_tokens = tokenizer.encode(context, return_tensors="pt")
            context_len = context_tokens.shape[1]
    elif dataset_name == "winogender" and len(biased_prompts) > 0:
        # For WinoGender, find the position before the pronoun
        # We'll use the position where we measure bias (before pronoun prediction)
        # For simplicity, find "that" token position as proxy for pronoun position
        first_sent = biased_prompts[0]
        tokens = tokenizer.encode(first_sent, return_tensors="pt")
        # Find "that" token position (common before pronoun in WinoGender)
        that_token_id = tokenizer.encode(" that", add_special_tokens=False)
        if that_token_id:
            that_id = that_token_id[0]
            # Find position of "that" in the sentence
            for i in range(tokens.shape[1]):
                if tokens[0, i].item() == that_id:
                    context_len = i + 1  # Position after "that"
                    break
        # Fallback: use 80% of sentence length
        if context_len is None:
            context_len = max(1, int(tokens.shape[1] * 0.8))
    
    # Create bias metric function with target tokens for first example
    target_tokens = target_words_list[0] if target_words_list and len(target_words_list) > 0 and target_words_list[0] else None
    bias_metric_fn = create_bias_metric_fn(model, tokenizer, dataset_name, target_tokens)
    
    # Wrap to handle both tensor and float returns
    def wrapped_bias_metric_fn(logits):
        result = bias_metric_fn(logits)
        if isinstance(result, torch.Tensor):
            return result.item() if result.numel() == 1 else result.mean().item()
        return result
    
    # Scan all edges (using first example pair)
    print("Scanning all edges...")
    impact_scores = scan_all_edges(model, [biased_tokens[0]], [neutral_tokens[0]], wrapped_bias_metric_fn, context_len=context_len)
    
    # Save results
    results_file = output_dir / f"causal_patching_{dataset_name}.json"
    with open(results_file, "w") as f:
        json.dump(impact_scores, f, indent=2)
    
    # Create ranked list
    ranked_edges = sorted(impact_scores.items(), key=lambda x: x[1], reverse=True)
    
    csv_file = output_dir / f"causal_patching_{dataset_name}_ranked.csv"
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["hook_name", "impact_score"])
        for hook_name, score in ranked_edges:
            writer.writerow([hook_name, score])
    
    print(f"Top 10 edges:")
    for hook_name, score in ranked_edges[:10]:
        print(f"  {hook_name}: {score:.4f}")
    
    return {
        "impact_scores": impact_scores,
        "ranked_edges": ranked_edges[:20]  # Top 20
    }


def run_attribution_patching_experiment(
    model,
    examples: List[Dict[str, Any]],
    dataset_name: str,
    tokenizer,
    output_dir: Path
) -> Dict[str, Any]:
    """
    Run attribution patching experiment.
    
    Args:
        model: HookedTransformer model
        examples: List of dataset examples
        dataset_name: Name of dataset
        tokenizer: Tokenizer instance
        output_dir: Directory to save results
    
    Returns:
        Dictionary of results
    """
    print(f"Running attribution patching experiment on {dataset_name}...")
    
    # Prepare examples and extract target tokens
    target_tokens_attr = None
    if dataset_name == "stereoset":
        pairs = get_stereoset_pairs(examples)
        prompts = []
        for p in pairs[:10]:
            context = p["context"]
            prompts.append(context)  # Use context for attribution
            # Extract target tokens from first pair
            if target_tokens_attr is None:
                stereo_sent = p["stereotype_sentence"]
                antistereo_sent = p["antistereotype_sentence"]
                context_prefix = context.replace("BLANK", "").strip()
                stereo_clean = stereo_sent.replace(context_prefix, "").strip()
                antistereo_clean = antistereo_sent.replace(context_prefix, "").strip()
                stereo_words = stereo_clean.split()
                antistereo_words = antistereo_clean.split()
                for sw, aw in zip(stereo_words, antistereo_words):
                    if sw != aw:
                        stereo_word = sw.rstrip('.,!?;:')
                        antistereo_word = aw.rstrip('.,!?;:')
                        stereo_tokens = tokenizer.encode(stereo_word, add_special_tokens=False)
                        antistereo_tokens = tokenizer.encode(antistereo_word, add_special_tokens=False)
                        if stereo_tokens and antistereo_tokens:
                            target_tokens_attr = {"stereotype": stereo_tokens, "antistereotype": antistereo_tokens}
                            break
    else:
        # For WinoGender, create pairs and extract pronouns
        prompts = []
        profession_groups = {}
        for ex in examples[:50]:
            prof = ex.get("profession", "")
            if prof and prof not in profession_groups:
                profession_groups[prof] = {"male": [], "female": []}
            answer = ex.get("answer", "").lower()
            if answer == "male" or ex.get("pronoun", "").lower() in ["he", "him", "his"]:
                profession_groups[prof]["male"].append(ex)
            elif answer == "female" or ex.get("pronoun", "").lower() in ["she", "her", "hers"]:
                profession_groups[prof]["female"].append(ex)
        
        for prof, groups in profession_groups.items():
            if groups["male"]:
                # Use sentence up to pronoun position
                sent = groups["male"][0]["sentence"]
                # Find "that" position as context end
                tokens = tokenizer.encode(sent, return_tensors="pt")
                that_token_id = tokenizer.encode(" that", add_special_tokens=False)
                if that_token_id:
                    that_id = that_token_id[0]
                    for i in range(tokens.shape[1]):
                        if tokens[0, i].item() == that_id:
                            context_tokens = tokens[:, :i+1]
                            context_text = tokenizer.decode(context_tokens[0])
                            prompts.append(context_text)
                            break
                    else:
                        prompts.append(sent)
                else:
                    prompts.append(sent)
                
                if target_tokens_attr is None and groups["male"] and groups["female"]:
                    male_pronoun = groups["male"][0].get("pronoun", "he")
                    female_pronoun = groups["female"][0].get("pronoun", "she")
                    male_tokens = tokenizer.encode(male_pronoun, add_special_tokens=False)
                    female_tokens = tokenizer.encode(female_pronoun, add_special_tokens=False)
                    if male_tokens and female_tokens:
                        target_tokens_attr = {"stereotype": male_tokens, "antistereotype": female_tokens}
                
                if len(prompts) >= 10:
                    break
        
        if not prompts:
            prompts = [ex["sentence"] for ex in examples[:10]]
    
    # Tokenize
    tokenized = [tokenizer.encode(p, return_tensors="pt").squeeze(0) for p in prompts]
    
    # Create bias metric function (needs gradients for attribution)
    bias_metric_fn = create_bias_metric_fn(model, tokenizer, dataset_name, target_tokens=target_tokens_attr)
    
    # Compute attributions
    print("Computing attributions...")
    attributions = attribution_patch(model, tokenized, bias_metric_fn)
    
    # Save results
    results_file = output_dir / f"attribution_patching_{dataset_name}.json"
    with open(results_file, "w") as f:
        json.dump(attributions, f, indent=2)
    
    # Get top edges
    ranked_edges = sorted(attributions.items(), key=lambda x: x[1], reverse=True)
    top_edges = ranked_edges[:20]
    
    # Validate top edges with causal patching
    print("Validating top edges with causal patching...")
    if dataset_name == "stereoset":
        pairs = get_stereoset_pairs(examples)
        biased_prompts = [p["context"] + " " + p["stereotype_sentence"] for p in pairs[:10]]
        neutral_prompts = [p["context"] + " " + p["antistereotype_sentence"] for p in pairs[:10]]
        biased_tokens = [tokenizer.encode(p, return_tensors="pt").squeeze(0) for p in biased_prompts]
        neutral_tokens = [tokenizer.encode(p, return_tensors="pt").squeeze(0) for p in neutral_prompts]
        # Get context length
        if len(pairs) > 0:
            context = pairs[0]["context"]
            context_tokens = tokenizer.encode(context, return_tensors="pt")
            context_len = context_tokens.shape[1]
        else:
            context_len = None
    else:
        # For WinoGender, create male/female pairs
        profession_groups = {}
        for ex in examples[:50]:
            prof = ex.get("profession", "")
            if prof and prof not in profession_groups:
                profession_groups[prof] = {"male": [], "female": []}
            answer = ex.get("answer", "").lower()
            if answer == "male" or ex.get("pronoun", "").lower() in ["he", "him", "his"]:
                profession_groups[prof]["male"].append(ex)
            elif answer == "female" or ex.get("pronoun", "").lower() in ["she", "her", "hers"]:
                profession_groups[prof]["female"].append(ex)
        
        biased_prompts = []
        neutral_prompts = []
        for prof, groups in profession_groups.items():
            if groups["male"] and groups["female"]:
                biased_prompts.append(groups["male"][0]["sentence"])
                neutral_prompts.append(groups["female"][0]["sentence"])
                if len(biased_prompts) >= 10:
                    break
        
        if not biased_prompts:
            biased_prompts = [ex["sentence"] for ex in examples[:10]]
            neutral_prompts = [ex["sentence"] for ex in examples[:10]]
        
        biased_tokens = [tokenizer.encode(p, return_tensors="pt").squeeze(0) for p in biased_prompts]
        neutral_tokens = [tokenizer.encode(p, return_tensors="pt").squeeze(0) for p in neutral_prompts]
        # Get context length (position before pronoun)
        if len(biased_prompts) > 0:
            first_sent = biased_prompts[0]
            tokens = tokenizer.encode(first_sent, return_tensors="pt")
            that_token_id = tokenizer.encode(" that", add_special_tokens=False)
            if that_token_id:
                that_id = that_token_id[0]
                for i in range(tokens.shape[1]):
                    if tokens[0, i].item() == that_id:
                        context_len = i + 1
                        break
                else:
                    context_len = max(1, int(tokens.shape[1] * 0.8))
            else:
                context_len = max(1, int(tokens.shape[1] * 0.8))
        else:
            context_len = None
    
    # Wrap bias metric for validation
    def wrapped_validation_metric(logits):
        result = bias_metric_fn(logits)
        if isinstance(result, torch.Tensor):
            return result.item() if result.numel() == 1 else result.mean().item()
        return result
    
    # Pass context_len for proper patching
    validation_context_len = context_len if 'context_len' in locals() else None
    validated = validate_top_edges(model, top_edges, biased_tokens, neutral_tokens, wrapped_validation_metric, context_len=validation_context_len)
    
    validated_file = output_dir / f"attribution_validated_{dataset_name}.json"
    with open(validated_file, "w") as f:
        json.dump(validated, f, indent=2)
    
    print(f"Top 10 attribution edges:")
    for hook_name, score in ranked_edges[:10]:
        print(f"  {hook_name}: {score:.4f}")
    
    return {
        "attributions": attributions,
        "validated": validated,
        "ranked_edges": ranked_edges[:20]
    }


def run_ablation_experiment(
    model,
    examples: List[Dict[str, Any]],
    dataset_name: str,
    tokenizer,
    output_dir: Path
) -> Dict[str, Any]:
    """
    Run head and MLP ablation experiment.
    
    Args:
        model: HookedTransformer model
        examples: List of dataset examples
        dataset_name: Name of dataset
        tokenizer: Tokenizer instance
        output_dir: Directory to save results
    
    Returns:
        Dictionary of results
    """
    print(f"Running ablation experiment on {dataset_name}...")
    
    # Prepare examples
    if dataset_name == "stereoset":
        pairs = get_stereoset_pairs(examples)
        prompts = [p["context"] + " " + p["stereotype_sentence"] for p in pairs[:10]]
    else:
        prompts = [ex["sentence"] for ex in examples[:10]]
    
    # Tokenize
    tokenized = [tokenizer.encode(p, return_tensors="pt").squeeze(0) for p in prompts]
    
    # Create bias metric function
    bias_metric_fn = create_bias_metric_fn(model, tokenizer, dataset_name, target_tokens=None)
    
    # Wrap to handle tensor returns
    def wrapped_bias_metric_fn(logits):
        result = bias_metric_fn(logits)
        if isinstance(result, torch.Tensor):
            return result.item() if result.numel() == 1 else result.mean().item()
        return result
    
    # Scan all heads
    print("Scanning all attention heads...")
    head_impacts = scan_all_heads(model, tokenized, wrapped_bias_metric_fn)
    
    # Scan all MLPs
    print("Scanning all MLPs...")
    mlp_impacts = scan_all_mlps(model, tokenized, wrapped_bias_metric_fn)
    
    # Save results
    head_file = output_dir / f"head_ablations_{dataset_name}.json"
    with open(head_file, "w") as f:
        # Convert tuple keys to strings for JSON
        head_dict = {f"{layer}_{head}": score for (layer, head), score in head_impacts.items()}
        json.dump(head_dict, f, indent=2)
    
    mlp_file = output_dir / f"mlp_ablations_{dataset_name}.json"
    with open(mlp_file, "w") as f:
        json.dump(mlp_impacts, f, indent=2)
    
    # Rank heads
    ranked_heads = sorted(head_impacts.items(), key=lambda x: abs(x[1]), reverse=True)
    ranked_mlps = sorted(mlp_impacts.items(), key=lambda x: abs(x[1]), reverse=True)
    
    print(f"Top 10 biased heads:")
    for (layer, head), impact in ranked_heads[:10]:
        print(f"  Layer {layer}, Head {head}: {impact:.4f}")
    
    print(f"Top 10 biased MLPs:")
    for layer, impact in ranked_mlps[:10]:
        print(f"  Layer {layer}: {impact:.4f}")
    
    # Convert tuple keys to strings for JSON serialization
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
    
    # Setup
    device = setup_device()
    model = load_model()
    model.to(device)
    tokenizer = get_tokenizer()
    
    # Create output directory
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    
    # Load datasets
    print("Loading datasets...")
    stereoset_examples = load_stereoset()
    winogender_examples = load_winogender()
    
    print(f"Loaded {len(stereoset_examples)} StereoSet examples")
    print(f"Loaded {len(winogender_examples)} WinoGender examples")
    
    # Compute baseline bias metrics
    print("\nComputing baseline bias metrics...")
    stereoset_baseline = compute_bias_metric(model, stereoset_examples, "stereoset", tokenizer)
    winogender_baseline = compute_bias_metric(model, winogender_examples, "winogender", tokenizer)
    
    print(f"StereoSet baseline bias: {stereoset_baseline:.4f}")
    print(f"WinoGender baseline bias: {winogender_baseline:.4f}")
    
    # Run experiments for each dataset
    datasets = [
        ("stereoset", stereoset_examples),
        ("winogender", winogender_examples)
    ]
    
    all_results = {}
    
    for dataset_name, examples in datasets:
        print(f"\n{'='*60}")
        print(f"Running experiments on {dataset_name}")
        print(f"{'='*60}")
        
        # Causal patching
        causal_results = run_causal_patching_experiment(
            model, examples, dataset_name, tokenizer, output_dir
        )
        
        # Attribution patching
        attribution_results = run_attribution_patching_experiment(
            model, examples, dataset_name, tokenizer, output_dir
        )
        
        # Ablations
        ablation_results = run_ablation_experiment(
            model, examples, dataset_name, tokenizer, output_dir
        )
        
        all_results[dataset_name] = {
            "baseline_bias": stereoset_baseline if dataset_name == "stereoset" else winogender_baseline,
            "causal_patching": causal_results,
            "attribution_patching": attribution_results,
            "ablations": ablation_results
        }
    
    # Save summary
    summary_file = output_dir / "summary.json"
    with open(summary_file, "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("Experiments complete! Results saved to", output_dir)
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

