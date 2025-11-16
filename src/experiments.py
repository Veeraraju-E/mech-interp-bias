"""Main experiment orchestration for bias analysis."""

import json
import torch
from typing import Dict, List, Any
from pathlib import Path
import torch.nn.functional as F
from tqdm import tqdm

from .model_setup import load_model, get_tokenizer, setup_device
from .data_loader import load_stereoset, load_winogender, get_stereoset_pairs
from .bias_metrics import compute_bias_metric
from .methods.activation_patching import attribution_patch, scan_all_heads, scan_all_mlps


def create_bias_metric_fn(model, tokenizer, dataset_name: str, target_tokens: Dict[str, List[int]] = None):
    """Create a bias metric function for use in patching experiments."""
    def bias_metric_fn(logits: torch.Tensor) -> torch.Tensor:
        if len(logits.shape) == 3:
            next_token_logits = logits[0, -1, :]
        else:
            next_token_logits = logits[-1, :]
        
        probs = F.softmax(next_token_logits, dim=-1)
        
        if dataset_name == "stereoset":
            if target_tokens and "stereotype" in target_tokens and "antistereotype" in target_tokens:
                stereo_token_ids = target_tokens["stereotype"]
                antistereo_token_ids = target_tokens["antistereotype"]
                
                stereo_prob = None
                for tid in stereo_token_ids:
                    if tid < len(probs):
                        stereo_prob = probs[tid] if stereo_prob is None else stereo_prob + probs[tid]
                
                if stereo_prob is None:
                    stereo_prob = torch.tensor(0.0, device=logits.device, requires_grad=True)
                
                antistereo_prob = None
                for tid in antistereo_token_ids:
                    if tid < len(probs):
                        antistereo_prob = probs[tid] if antistereo_prob is None else antistereo_prob + probs[tid]
                
                if antistereo_prob is None:
                    antistereo_prob = torch.tensor(0.0, device=logits.device, requires_grad=True)
                
                return torch.log(stereo_prob + 1e-10) - torch.log(antistereo_prob + 1e-10)
            else:
                entropy = -(probs * torch.log(probs + 1e-10)).sum()
                return -entropy
        
        elif dataset_name == "winogender":
            if len(logits.shape) == 3:
                next_token_logits = logits[0, -1, :]
            else:
                next_token_logits = logits[-1, :]
            
            probs = F.softmax(next_token_logits, dim=-1)
            male_pronouns = ["he", "him", "his"]
            female_pronouns = ["she", "her", "hers"]
            
            male_prob = None
            for p in male_pronouns:
                p_tokens = tokenizer.encode(p, add_special_tokens=False)
                if len(p_tokens) > 0 and p_tokens[0] < len(probs):
                    male_prob = probs[p_tokens[0]] if male_prob is None else male_prob + probs[p_tokens[0]]
            
            if male_prob is None:
                male_prob = torch.tensor(0.0, device=logits.device, requires_grad=True)
            
            female_prob = None
            for p in female_pronouns:
                p_tokens = tokenizer.encode(p, add_special_tokens=False)
                if len(p_tokens) > 0 and p_tokens[0] < len(probs):
                    female_prob = probs[p_tokens[0]] if female_prob is None else female_prob + probs[p_tokens[0]]
            
            if female_prob is None:
                female_prob = torch.tensor(0.0, device=logits.device, requires_grad=True)
            
            return male_prob - female_prob
        
        else:
            return logits.mean()
    
    return bias_metric_fn

def run_attribution_patching_experiment(
    model,
    examples: List[Dict[str, Any]],
    dataset_name: str,
    tokenizer,
    output_dir: Path
) -> Dict[str, Any]:
    """Run attribution patching experiment."""
    print(f"Running attribution patching experiment on {dataset_name}...")
    
    target_tokens_attr = None
    if dataset_name == "stereoset":
        pairs = get_stereoset_pairs(examples)
        prompts = []
        for p in pairs:
            context = p["context"]
            prompts.append(context)
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
        prompts = []
        profession_groups = {}
        for ex in examples:
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
                sent = groups["male"][0]["sentence"]
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
        
        if not prompts:
            prompts = [ex["sentence"] for ex in examples]
    
    tokenized = [tokenizer.encode(p, return_tensors="pt").squeeze(0) for p in prompts]
    bias_metric_fn = create_bias_metric_fn(model, tokenizer, dataset_name, target_tokens=target_tokens_attr)
    
    print("Computing attributions...")
    attributions = attribution_patch(model, tokenized, bias_metric_fn)
    
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
    examples: List[Dict[str, Any]],
    dataset_name: str,
    tokenizer,
    output_dir: Path
) -> Dict[str, Any]:
    """Run head and MLP ablation experiment."""
    print(f"Running ablation experiment on {dataset_name}...")
    
    if dataset_name == "stereoset":
        pairs = get_stereoset_pairs(examples)
        prompts = [p["context"] + " " + p["stereotype_sentence"] for p in pairs]
    else:
        prompts = []
        for ex in examples:
            sent = ex.get("sentence", "")
            if not sent:
                continue
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
                    context_len = max(1, int(tokens.shape[1] * 0.8))
                    context_tokens = tokens[:, :context_len]
                    context_text = tokenizer.decode(context_tokens[0])
                    prompts.append(context_text)
            else:
                context_len = max(1, int(tokens.shape[1] * 0.8))
                context_tokens = tokens[:, :context_len]
                context_text = tokenizer.decode(context_tokens[0])
                prompts.append(context_text)
        
        if not prompts:
            prompts = [ex["sentence"] for ex in examples]
    
    tokenized = [tokenizer.encode(p, return_tensors="pt").squeeze(0) for p in prompts]
    bias_metric_fn = create_bias_metric_fn(model, tokenizer, dataset_name, target_tokens=None)
    
    def wrapped_bias_metric_fn(logits):
        result = bias_metric_fn(logits)
        if isinstance(result, torch.Tensor):
            return result.item() if result.numel() == 1 else result.mean().item()
        return result
    
    print("Scanning all attention heads...")
    head_impacts = scan_all_heads(model, tokenized, wrapped_bias_metric_fn)
    
    print("Scanning all MLPs...")
    mlp_impacts = scan_all_mlps(model, tokenized, wrapped_bias_metric_fn)
    
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
        
        attribution_results = run_attribution_patching_experiment(
            model, examples, dataset_name, tokenizer, output_dir
        )
        
        ablation_results = run_ablation_experiment(
            model, examples, dataset_name, tokenizer, output_dir
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

