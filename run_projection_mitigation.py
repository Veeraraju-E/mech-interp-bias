"""
This script runs two separate bias mitigation methods:
1. Projection-Based Steering: Orthogonal projection to remove bias
2. SAE-Based Projection Steering: Using SAE to find better bias direction

Usage:
    # Run both methods
    python run_projection_mitigation.py --dataset stereoset --method both --use-wandb
    
    # Run only projection-based
    python run_projection_mitigation.py --dataset stereoset --method projection --use-wandb
    
    # Run only SAE-based
    python run_projection_mitigation.py --dataset stereoset --method sae --use-wandb
"""

import argparse
import json
from pathlib import Path
import torch
import wandb

from src.model_setup import load_model, get_tokenizer
from src.data_loader import (
    load_stereoset, 
    load_winogender, 
    get_stereoset_pairs, 
    build_winogender_pairs,
    find_first_difference_tokens
)
from src.methods.bias_vectors import (
    extract_bias_vectors_all_layers,
    visualize_bias_vectors
)
from src.methods.bias_vectors.projection_steering import (
    batch_projection_steering_experiment,
    visualize_projection_steering_results,
    generate_projection_steering_examples
)
from src.methods.bias_vectors.sae_steering import (
    batch_sae_steering_experiment
)


def get_bias_critical_layers(model_name: str) -> list:
    """
    Get bias-critical layers based on model architecture.
    
    From requirements: layers 11-18 for GPT-2 Medium, 25-31 for Large.
    """
    if "medium" in model_name.lower():
        return list(range(11, 19))  # 11-18
    elif "large" in model_name.lower():
        return list(range(25, 32))  # 25-31
    else:
        # For small models, use later layers
        n_layers = 12  # GPT-2 small default
        return list(range(max(0, n_layers - 7), n_layers))


def prepare_bias_prompts(dataset_name: str, data_dir: Path = Path("data")):
    """Prepare biased and neutral prompts from datasets."""
    if dataset_name == "stereoset":
        examples = load_stereoset(data_dir)
        pairs = get_stereoset_pairs(examples)
        
        # Filter for gender bias
        gender_pairs = [p for p in pairs if p.get("bias_type") == "gender"]
        
        biased_prompts = []
        neutral_prompts = []
        steering_prompts = []
        
        for pair in gender_pairs[:200]:
            # Biased: stereotype
            biased_prompts.append(pair["stereotype_sentence"])
            # Neutral: antistereotype
            neutral_prompts.append(pair["antistereotype_sentence"])
            
            # For steering - extract ACTUAL difference words
            prefix = pair.get("context_prefix", "")
            if prefix and len(steering_prompts) < 200:
                # Find actual difference words between stereotype and antistereotype
                diff_words = find_first_difference_tokens(
                    pair.get("stereotype_sentence", ""),
                    pair.get("antistereotype_sentence", "")
                )
                
                if diff_words and diff_words[0] and diff_words[1]:
                    # Use actual difference words
                    stereo_word = diff_words[0]
                    anti_word = diff_words[1]
                    steering_prompts.append({
                        "text": prefix,
                        "biased_tokens": [stereo_word],
                        "neutral_tokens": [anti_word]
                    })
                else:
                    # Fallback to pronouns if difference words not found
                    steering_prompts.append({
                        "text": prefix,
                        "biased_tokens": ["he", "him", "his", "man", "male"],
                        "neutral_tokens": ["she", "her", "hers", "woman", "female"]
                    })
    
    elif dataset_name == "winogender":
        examples = load_winogender(data_dir)
        pairs = build_winogender_pairs(examples)
        
        biased_prompts = []
        neutral_prompts = []
        steering_prompts = []
        
        for pair in pairs[:200]:
            biased_prompts.append(pair.get("male_sentence", ""))
            neutral_prompts.append(pair.get("female_sentence", ""))
            
            prompt = pair.get("prompt", "")
            if prompt and len(steering_prompts) < 200:
                steering_prompts.append({
                    "text": prompt,
                    "biased_tokens": ["he", "him", "his"],
                    "neutral_tokens": ["she", "her", "hers"]
                })
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return biased_prompts, neutral_prompts, steering_prompts


def main():
    parser = argparse.ArgumentParser(description="Run projection-based bias mitigation")
    parser.add_argument("--dataset", type=str, default="stereoset", choices=["stereoset", "winogender"])
    parser.add_argument("--model", type=str, default="gpt2-medium",
                       choices=["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl",
                               "gpt-neo-125m", "gpt-neo-1.3b", "gpt-neo-2.7b"],
                       help="Model to use for analysis")
    parser.add_argument("--method", type=str, default="both", choices=["projection", "sae", "both"],
                        help="Which mitigation method to run")
    parser.add_argument("--output-dir", type=str, default="results/projection_mitigation")
    parser.add_argument("--target-layer", type=int, default=None, help="Target layer (default: auto-detect)")
    parser.add_argument("--use-bias-critical-layers", action="store_true",
                        help="Test multiple bias-critical layers")
    parser.add_argument("--skip-extraction", action="store_true", help="Skip vector extraction")
    parser.add_argument("--use-wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default="mech-interp-bias", help="W&B project name")
    parser.add_argument("--wandb-entity", type=str, default=None, help="W&B entity/username")
    
    # SAE-specific arguments
    parser.add_argument("--sae-n-features", type=int, default=512, help="Number of SAE features")
    parser.add_argument("--sae-sparsity", type=float, default=0.01, help="SAE sparsity coefficient")
    parser.add_argument("--sae-epochs", type=int, default=100, help="SAE training epochs")
    parser.add_argument("--sae-batch-size", type=int, default=256, help="SAE training batch size")
    parser.add_argument("--sae-lr", type=float, default=1e-3, help="SAE learning rate")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize W&B if requested
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=f"projection_mitigation_{args.dataset}_{args.method}",
            config={
                "dataset": args.dataset,
                "model": args.model,
                "method": args.method,
                "target_layer": args.target_layer,
                "use_bias_critical_layers": args.use_bias_critical_layers,
                "sae_n_features": args.sae_n_features,
                "sae_sparsity": args.sae_sparsity,
                "sae_epochs": args.sae_epochs,
                "section": "4.1_4.2_projection_mitigation"
            },
            tags=["projection_steering", "sae_steering", "bias_mitigation", args.dataset]
        )
        print("✅ Weights & Biases initialized!")
    
    print(f"\n{'='*70}")
    print("Section 4.1 & 4.2: Projection-Based Bias Mitigation")
    print(f"{'='*70}")
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model}")
    print(f"Method: {args.method}")
    print(f"Output directory: {output_dir}")
    print(f"W&B logging: {args.use_wandb}")
    print(f"{'='*70}\n")
    
    # Load model
    print("Loading model...")
    model = load_model(args.model)
    tokenizer = get_tokenizer(args.model)
    n_layers = model.cfg.n_layers
    
    # Prepare data
    print(f"\nPreparing {args.dataset} data...")
    biased_prompts, neutral_prompts, steering_prompts = prepare_bias_prompts(args.dataset)
    
    print(f"Biased prompts: {len(biased_prompts)}")
    print(f"Neutral prompts: {len(neutral_prompts)}")
    print(f"Steering prompts: {len(steering_prompts)}")
    
    # Extract bias vectors (mean difference) for projection-based steering
    if not args.skip_extraction:
        print(f"\n{'='*70}")
        print("Extracting Mean Difference Bias Vectors")
        print(f"{'='*70}\n")
        
        vectors_dict = extract_bias_vectors_all_layers(
            model=model,
            biased_prompts=biased_prompts,
            neutral_prompts=neutral_prompts,
            tokenizer=tokenizer,
            enhance=True
        )
        
        # Save vectors
        vectors_file = output_dir / f"bias_vectors_{args.dataset}.pt"
        torch.save(vectors_dict, vectors_file)
        print(f"\nBias vectors saved to: {vectors_file}")
        
        # Determine target layer
        if args.target_layer is None:
            target_layer = max(vectors_dict.keys(), 
                             key=lambda l: vectors_dict[l].get('enhanced_norm', vectors_dict[l]['norm']))
            print(f"\nAuto-detected target layer: {target_layer}")
        else:
            target_layer = args.target_layer
        
        # Extract bias vectors for steering
        bias_vectors = {}
        for layer, data in vectors_dict.items():
            bias_vectors[layer] = data.get('enhanced_vector', data['vector'])
    else:
        # Load pre-extracted vectors
        vectors_file = output_dir / f"bias_vectors_{args.dataset}.pt"
        if vectors_file.exists():
            vectors_dict = torch.load(vectors_file)
            print(f"Loaded bias vectors from: {vectors_file}")
            target_layer = args.target_layer if args.target_layer is not None else n_layers // 2
            
            bias_vectors = {}
            for layer, data in vectors_dict.items():
                bias_vectors[layer] = data.get('enhanced_vector', data['vector'])
        else:
            print("No pre-extracted vectors found. Please run without --skip-extraction first.")
            return
    
    # Determine layers to test
    if args.use_bias_critical_layers:
        bias_critical_layers = get_bias_critical_layers(args.model)
        # Filter to only layers that have bias vectors
        bias_critical_layers = [l for l in bias_critical_layers if l in bias_vectors]
        if not bias_critical_layers:
            bias_critical_layers = [target_layer]
        print(f"\nTesting bias-critical layers: {bias_critical_layers}")
    else:
        bias_critical_layers = None
    
    # Part 1: Projection-Based Steering (Section 4.1)
    if args.method in ["projection", "both"]:
        print(f"\n{'='*70}")
        print("Part 1: Projection-Based Steering (Section 4.1)")
        print(f"{'='*70}\n")
        
        projection_results = batch_projection_steering_experiment(
            model=model,
            prompts=steering_prompts,
            tokenizer=tokenizer,
            bias_vectors=bias_vectors,
            target_layer=target_layer,
            bias_critical_layers=bias_critical_layers,
            hook_name="hook_resid_post",
            position=-1
        )
        
        # Visualize
        print("\nVisualizing projection steering results...")
        visualize_projection_steering_results(
            projection_results,
            output_dir=output_dir / "projection_steering",
            save_prefix=f"projection_{args.dataset}"
        )
        
        # Generate examples
        print("\n" + "="*70)
        print("Generating Example Completions with Projection Steering")
        print("="*70 + "\n")
        
        example_prompts = [p["text"] for p in steering_prompts[:5]]
        from src.methods.bias_vectors.projection_steering import generate_projection_steering_examples
        examples = generate_projection_steering_examples(
            model=model,
            prompts=example_prompts,
            tokenizer=tokenizer,
            bias_vector=bias_vectors[target_layer],
            layer_idx=target_layer,
            max_new_tokens=15,
            output_file=output_dir / "projection_steering" / f"examples_{args.dataset}.json"
        )
        
        # Log to W&B
        if args.use_wandb:
            for layer, result in projection_results.items():
                wandb.log({
                    f"projection_steering/layer_{layer}/baseline_bias": result['baseline_bias_mean'],
                    f"projection_steering/layer_{layer}/projected_bias": result['projected_bias_mean'],
                    f"projection_steering/layer_{layer}/reduction": result['mean_reduction'],
                    f"projection_steering/layer_{layer}/relative_reduction": result['relative_reduction'],
                    "layer": layer
                })
            
            # Log visualization
            plot_path = output_dir / "projection_steering" / f"projection_{args.dataset}_comparison.png"
            if plot_path.exists():
                wandb.log({"projection_steering/comparison_plot": wandb.Image(str(plot_path))})
            
            print("✅ Projection steering results logged to W&B")
    
    # Part 2: SAE-Based Projection Steering (Section 4.2)
    if args.method in ["sae", "both"]:
        print(f"\n{'='*70}")
        print("Part 2: SAE-Based Projection Steering (Section 4.2)")
        print(f"{'='*70}\n")
        
        # Determine layers for SAE (can be expensive, so maybe just target layer)
        sae_layers = [target_layer] if not args.use_bias_critical_layers else bias_critical_layers[:3]  # Limit to 3 for speed
        
        sae_results = batch_sae_steering_experiment(
            model=model,
            prompts=steering_prompts,
            tokenizer=tokenizer,
            biased_prompts=biased_prompts,
            neutral_prompts=neutral_prompts,
            target_layer=target_layer,
            bias_critical_layers=sae_layers,
            hook_name="hook_resid_post",
            position=-1,
            n_features=args.sae_n_features,
            sparsity_coeff=args.sae_sparsity,
            n_epochs=args.sae_epochs,
            batch_size=args.sae_batch_size,
            lr=args.sae_lr
        )
        
        # Visualize SAE results
        print("\nVisualizing SAE steering results...")
        sae_results_dict = sae_results["results"]
        if sae_results_dict:
            visualize_projection_steering_results(
                sae_results_dict,
                output_dir=output_dir / "sae_steering",
                save_prefix=f"sae_{args.dataset}"
            )
        
        # Save SAE vectors
        sae_vectors_file = output_dir / "sae_steering" / f"sae_vectors_{args.dataset}.pt"
        sae_vectors_file.parent.mkdir(parents=True, exist_ok=True)
        torch.save(sae_results["sae_vectors"], sae_vectors_file)
        print(f"\nSAE vectors saved to: {sae_vectors_file}")
        
        # Log to W&B
        if args.use_wandb:
            for layer, result in sae_results_dict.items():
                wandb.log({
                    f"sae_steering/layer_{layer}/baseline_bias": result['baseline_bias_mean'],
                    f"sae_steering/layer_{layer}/projected_bias": result['projected_bias_mean'],
                    f"sae_steering/layer_{layer}/reduction": result['mean_reduction'],
                    f"sae_steering/layer_{layer}/relative_reduction": result['relative_reduction'],
                    f"sae_steering/layer_{layer}/sae_vector_norm": result['sae_vector_norm'],
                    "layer": layer
                })
            
            # Log visualization
            plot_path = output_dir / "sae_steering" / f"sae_{args.dataset}_comparison.png"
            if plot_path.exists():
                wandb.log({"sae_steering/comparison_plot": wandb.Image(str(plot_path))})
            
            print("✅ SAE steering results logged to W&B")
    
    print(f"\n{'='*70}")
    print("Projection-based bias mitigation complete!")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*70}\n")
    
    # Finish W&B run
    if args.use_wandb:
        wandb.finish()
        print("✅ W&B run finished")


if __name__ == "__main__":
    main()

