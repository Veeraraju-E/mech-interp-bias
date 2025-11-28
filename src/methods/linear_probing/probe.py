"""Linear probes: train logistic regression on activations to predict bias."""
import os
import sys
import argparse
import torch
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from transformer_lens import HookedTransformer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model_setup import *
from src.data_loader import *
from src.cache_utils import *


class ProbeResults:
    """Results from linear probing experiment."""
    def __init__(self):
        self.layer_accuracies: Dict[int, float] = {}
        self.layer_aucs: Dict[int, float] = {}
        self.layer_probes: Dict[int, LogisticRegression] = {}
        self.activations: Dict[int, np.ndarray] = {}
        self.labels: np.ndarray = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert results to dictionary for JSON serialization."""
        return {
            'layer_accuracies': self.layer_accuracies,
            'layer_aucs': self.layer_aucs,
            'num_examples': len(self.labels) if self.labels is not None else 0
        }


def collect_activations(model: HookedTransformer, tokenized_prompts: List[torch.Tensor], layer_indices: Optional[List[int]] = None, position: str = "last") -> Dict[int, np.ndarray]:
    """Collect residual stream activations at each layer."""
    
    if layer_indices is None:
        layer_indices = list(range(model.cfg.n_layers + 1))
    
    activations = {layer: [] for layer in layer_indices}
    
    for tokens in tqdm(tokenized_prompts, desc="Collecting activations"):
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
        tokens = tokens.to(model.cfg.device)
        
        saved_activations = {layer: None for layer in layer_indices}
        
        def save_activation(layer_idx: int):
            def hook_fn(activation: torch.Tensor, hook):
                saved_activations[layer_idx] = activation.detach().cpu()
                return activation
            return hook_fn
        
        hooks = []
        for layer in layer_indices:
            hook_name = "blocks.0.hook_resid_pre" if layer == 0 else f"blocks.{layer-1}.hook_resid_post"
            hooks.append((hook_name, save_activation(layer)))
        
        with torch.no_grad():
            model.run_with_hooks(tokens, fwd_hooks=hooks)
        
        for layer in layer_indices:
            if saved_activations[layer] is not None:
                layer_act = saved_activations[layer]
                
                if layer_act.dim() == 3:
                    layer_act = layer_act[0]
                
                if position == "last":
                    vec = layer_act[-1, :].numpy()
                elif position == "mean":
                    vec = layer_act.mean(dim=0).numpy()
                else:
                    raise ValueError(f"Invalid position: {position}")
                
                activations[layer].append(vec)
    
    for layer in layer_indices: # stack along first dimension: [n_examples, hidden_dim]
        activations[layer] = np.stack(activations[layer], axis=0) if activations[layer] else np.array([])
    
    return activations


def prepare_biased_neutral_pairs(examples: Optional[List[Dict[str, Any]]], dataset_name: str, tokenizer) -> Tuple[List[str], np.ndarray]:
    """Prepare biased and neutral prompts with labels."""

    prompts = []
    labels = []
    
    if dataset_name in ["stereoset_race", "stereoset_gender"]:
        import json
        from pathlib import Path
        
        bias_type = "race" if dataset_name == "stereoset_race" else "gender"
        data_dir = Path("data")
        acdc_file = data_dir / f"stereoset_{bias_type}_acdc_pairs.json"
        
        with open(acdc_file, "r") as f:
            pairs = json.load(f)
        
        for pair in pairs:
            clean_entry = pair.get("clean", {})
            corrupted_entry = pair.get("corrupted", {})
            
            if clean_entry and clean_entry.get("tokens"):
                clean_tokens = torch.tensor(clean_entry["tokens"], dtype=torch.long)
                clean_metadata = clean_entry.get("metadata", {})
                clean_prompt = tokenizer.decode(clean_tokens)
                clean_label = 1 if clean_metadata.get("label") == "stereotype" else 0
                prompts.append(clean_prompt)
                labels.append(clean_label)
            
            if corrupted_entry and corrupted_entry.get("tokens"):
                corrupted_tokens = torch.tensor(corrupted_entry["tokens"], dtype=torch.long)
                corrupted_metadata = corrupted_entry.get("metadata", {})
                corrupted_prompt = tokenizer.decode(corrupted_tokens)
                corrupted_label = 1 if corrupted_metadata.get("label") == "stereotype" else 0
                prompts.append(corrupted_prompt)
                labels.append(corrupted_label)
    
    elif dataset_name == "stereoset":        
        pairs = get_stereoset_pairs(examples)
        for pair in pairs:
            context = pair["context"]
            stereo_prompt = context + " " + pair["stereotype_sentence"]
            antistereo_prompt = context + " " + pair["antistereotype_sentence"]
            
            prompts.append(stereo_prompt)
            labels.append(1)
            
            prompts.append(antistereo_prompt)
            labels.append(0)
    
    elif dataset_name == "winogender":
        from collections import defaultdict
        
        # Group examples by profession and word (the entity the pronoun refers to)
        profession_word_groups = defaultdict(lambda: {"male": [], "female": []})
        
        for ex in examples:
            sentence = ex.get("sentence", "")
            if not sentence: continue
            
            profession = ex.get("profession", "").strip().lower()
            if not profession: continue

            word = ex.get("word", "").strip().lower()
            key = f"{profession}_{word}" if word else profession
            
            answer = ex.get("answer", "").lower()
            pronoun = ex.get("pronoun", "").lower()

            if answer == "male" or pronoun in ["he", "him", "his"]:
                profession_word_groups[key]["male"].append(ex)
            elif answer == "female" or pronoun in ["she", "her", "hers"]:
                profession_word_groups[key]["female"].append(ex)
        
        # Create pairs: for each profession+word combination with both male and female examples
        for key, groups in profession_word_groups.items():
            if groups["male"] and groups["female"]:
                min_pairs = min(len(groups["male"]), len(groups["female"]))
                for i in range(min_pairs):
                    male_ex = groups["male"][i]
                    female_ex = groups["female"][i]
                    
                    male_sent = male_ex["sentence"]
                    female_sent = female_ex["sentence"]
                    
                    # Extract context up to "that" (where pronoun would be predicted)
                    male_tokens = tokenizer.encode(male_sent, return_tensors="pt")
                    female_tokens = tokenizer.encode(female_sent, return_tensors="pt")
                    
                    that_token_id = tokenizer.encode(" that", add_special_tokens=False)
                    if that_token_id:
                        that_id = that_token_id[0]
                        
                        # Find "that" position in male sentence
                        male_context_tokens = male_tokens
                        for j in range(male_tokens.shape[1]):
                            if male_tokens[0, j].item() == that_id:
                                male_context_tokens = male_tokens[:, :j+1]
                                break
                        
                        # Find "that" position in female sentence
                        female_context_tokens = female_tokens
                        for j in range(female_tokens.shape[1]):
                            if female_tokens[0, j].item() == that_id:
                                female_context_tokens = female_tokens[:, :j+1]
                                break
                        
                        male_prompt = tokenizer.decode(male_context_tokens[0])
                        female_prompt = tokenizer.decode(female_context_tokens[0])
                    else:
                        print(f'no token found for that, skipping example {i}')
                        continue
                    
                    prompts.append(male_prompt)
                    labels.append(1)
                    
                    prompts.append(female_prompt)
                    labels.append(0)
    
    labels = np.array(labels)
    return prompts, labels


def train_probe(activations: np.ndarray, labels: np.ndarray, train_indices: np.ndarray, test_indices: np.ndarray) -> Tuple[LogisticRegression, float, float]:
    """Train a logistic regression probe and evaluate."""

    X_train = activations[train_indices]
    y_train = labels[train_indices]
    X_test = activations[test_indices]
    y_test = labels[test_indices]
    
    probe = LogisticRegression(max_iter=1000, random_state=42)
    probe.fit(X_train, y_train)
    
    test_pred = probe.predict(X_test)
    test_probs = probe.predict_proba(X_test)[:, 1]
    
    test_acc = accuracy_score(y_test, test_pred)
    test_auc = roc_auc_score(y_test, test_probs)

    return probe, test_acc, test_auc

# main
def train_layer_probes(
    model: HookedTransformer,
    examples: Optional[List[Dict[str, Any]]],
    dataset_name: str,
    tokenizer,
    output_dir: Optional[Path] = None,
    train_split: float = 0.8,
    position: str = "last",
    cache_dir: Optional[Path] = None,
    use_cache: bool = True
) -> ProbeResults:
    """Train linear probes on each layer's residual stream activations."""

    print(f"Training linear probes for {dataset_name}...")
    
    model_name = get_model_name(model)
    prompts, labels = prepare_biased_neutral_pairs(examples, dataset_name, tokenizer)
    print(f"Prepared {len(prompts)} prompts ({np.sum(labels)} biased, {len(labels) - np.sum(labels)} neutral)")
    
    n_layers = model.cfg.n_layers
    layer_indices = list(range(n_layers + 1))
    
    activations_dict = None
    cached_labels = None
    
    if cache_dir and use_cache:
        cached_data = load_cached_activations(cache_dir, model_name, dataset_name, position)
        if cached_data is not None:
            activations_dict, cached_labels = cached_data
            if cached_labels is not None:
                labels = cached_labels
                print(f"Loaded cached activations for {len(labels)} examples")
    
    if activations_dict is None:
        tokenized = []
        for p in prompts:
            tokens = tokenizer.encode(p, return_tensors="pt").squeeze(0)
            tokenized.append(tokens)
        
        print("Collecting activations...")
        activations_dict = collect_activations(model, tokenized, layer_indices, position=position)
        
        if cache_dir and use_cache:
            cache_activations(cache_dir, model_name, dataset_name, activations_dict, labels, position, use_cache)
    
    n_examples = len(labels)
    indices = np.arange(n_examples)
    np.random.seed(42)
    np.random.shuffle(indices)
    
    split_idx = int(n_examples * train_split)
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]
    
    results = ProbeResults()
    results.labels = labels
    
    print("Training probes for each layer...")
    for layer in tqdm(layer_indices, desc="Training probes"):
        activation_vecs = activations_dict[layer]
        
        if len(activation_vecs) == 0:
            print(f"Warning: Layer {layer} has no activations")
            continue
        
        if len(activation_vecs) != n_examples:
            print(f"Warning: Layer {layer} has {len(activation_vecs)} examples, expected {n_examples}")
            continue
        
        results.activations[layer] = activation_vecs
        
        probe, acc, auc = train_probe(activation_vecs, labels, train_indices, test_indices)
        
        results.layer_probes[layer] = probe
        results.layer_accuracies[layer] = float(acc)
        results.layer_aucs[layer] = float(auc)
    
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        results_file = output_dir / f"linear_probing_{dataset_name}.json"
        
        import json
        with open(results_file, "w") as f:
            json.dump(results.to_dict(), f, indent=2)
    
    print("\nLayer probe accuracies:")
    for layer, acc in results.layer_accuracies.items():
        print(f"  Layer {layer}: Accuracy={acc:.4f}")
    
    return results


def main():
    """Run linear probing experiments for both datasets."""
    parser = argparse.ArgumentParser(description="Linear probing analysis for bias detection")
    parser.add_argument("--model", type=str, default="gpt2-medium", choices=["gpt2-medium", "gpt2-large", "gpt-neo-125M"], help="Model to use (default: gpt2-medium)")
    parser.add_argument("--no-cache", action="store_true", help="Disable caching and recompute everything from scratch")
    parser.add_argument("--cache-dir", type=str, default="runs/linear_probing/cache", help="Directory to store/load cached activations (default: runs/linear_probing/cache)")
    parser.add_argument("--output-dir", type=str, default="results", help="Directory to save results (default: results)")
    parser.add_argument("--position", type=str, default="last", choices=["last", "mean"], help="Position to extract activations from (default: last)")
    args = parser.parse_args()
    
    print("\nInitializing model...")
    print(f"Using model: {args.model}")
    device = setup_device()
    model = load_model(args.model)
    model.to(device)
    tokenizer = get_tokenizer(args.model)
    model_name = get_model_name(model)
    
    output_dir = Path(args.output_dir) / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cache_dir = Path(args.cache_dir) if not args.no_cache else None
    use_cache = not args.no_cache
    
    if cache_dir:
        print(f"Cache directory: {cache_dir}")
        if use_cache:
            print("Caching enabled")
        else:
            print("Caching disabled")
    
    print("Loading datasets...")
    stereoset_examples = load_stereoset()
    winogender_examples = load_winogender()
    
    print(f"Loaded {len(stereoset_examples)} StereoSet examples")
    print(f"Loaded {len(winogender_examples)} WinoGender examples")
    
    datasets = [("stereoset_race", None), ("stereoset_gender", None), ("winogender", winogender_examples)]
    
    all_results = {}
    
    for dataset_name, examples in datasets:
        print(f"\n{'='*60}")
        print(f"Running linear probing on {dataset_name}")
        print(f"{'='*60}")

        results = train_layer_probes(model=model, examples=examples, dataset_name=dataset_name, tokenizer=tokenizer, output_dir=output_dir, train_split=0.8, position=args.position, cache_dir=cache_dir, use_cache=use_cache)
        results_dict =  {
            "layer_accuracies": results.layer_accuracies,
            "layer_aucs": results.layer_aucs,
            "num_examples": len(results.labels) if results.labels is not None else 0,
            "ranked_layers": sorted(results.layer_accuracies.items(), key=lambda x: x[1], reverse=True)
        }

        all_results[dataset_name] = results_dict
    
    summary_file = output_dir / "linear_probing_summary.json"
    with open(summary_file, "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("Linear probing experiments complete!")
    print(f"Results saved to {output_dir}")
    if cache_dir and use_cache:
        print(f"Cache saved to {cache_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()