"""
This script demonstrates:
1. Mean difference vectors
2. Enhanced vectors with PCA validation
3. Steering experiments
4. Gradient-based vector search (novel contribution)

Usage:
    python run_bias_vector_analysis.py --dataset stereoset --model gpt2-medium
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
    find_first_difference_tokens,
    load_acdc_pairs,
    get_acdc_stereoset_pairs,
    get_acdc_winogender_pairs
)
from src.methods.bias_vectors import (
    extract_bias_vectors_all_layers,
    visualize_bias_vectors,
    batch_steering_experiment,
    visualize_steering_results,
    generate_steering_examples,
    optimize_bias_direction,
    visualize_gradient_search
)


def prepare_bias_prompts(dataset_name: str, data_dir: Path = Path("data"), use_acdc: bool = False):
    """Prepare biased and neutral prompts from datasets.
    
    Args:
        dataset_name: Dataset to load
        data_dir: Directory containing data files
        use_acdc: If True, use ACDC-formatted pairs
    """
    if use_acdc:
        # Use new ACDC pair format
        if dataset_name.startswith("stereoset"):
            if dataset_name == "stereoset":
                pairs = get_acdc_stereoset_pairs("gender", data_dir)
            else:
                bias_type = dataset_name.split("_")[1]
                pairs = get_acdc_stereoset_pairs(bias_type, data_dir)
            
            biased_prompts = []
            neutral_prompts = []
            steering_prompts = []
            
            for pair in pairs[:200]:
                clean = pair.get("clean", {})
                corrupted = pair.get("corrupted", {})
                metadata = clean.get("metadata", {})
                
                label = metadata.get("label", "stereotype")
                if label == "stereotype":
                    biased_prompts.append(clean["sentence"])
                    neutral_prompts.append(corrupted["sentence"])
                else:
                    biased_prompts.append(corrupted["sentence"])
                    neutral_prompts.append(clean["sentence"])
                
                context = clean.get("context", "")
                if context and len(steering_prompts) < 200:
                    steering_prompts.append({
                        "text": context,
                        "biased_tokens": [metadata.get("stereotype_word", "he")],
                        "neutral_tokens": [metadata.get("antistereotype_word", "she")]
                    })
        
        elif dataset_name == "winogender":
            pairs = get_acdc_winogender_pairs(data_dir)
            
            biased_prompts = []
            neutral_prompts = []
            steering_prompts = []
            
            for pair in pairs[:200]:
                clean = pair.get("clean", {})
                corrupted = pair.get("corrupted", {})
                metadata = clean.get("metadata", {})
                
                biased_prompts.append(corrupted["sentence"])
                neutral_prompts.append(clean["sentence"])
                
                context = clean.get("context", "")
                if context and len(steering_prompts) < 200:
                    steering_prompts.append({
                        "text": context,
                        "biased_tokens": ["he", "him", "his"],
                        "neutral_tokens": ["she", "her", "hers"],
                        "profession": metadata.get("profession", "")
                    })
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
    
    elif dataset_name == "stereoset":
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
    parser = argparse.ArgumentParser(description="Run bias vector identification and steering")
    parser.add_argument("--dataset", type=str, default="stereoset", 
                       choices=["stereoset", "stereoset_gender", "stereoset_race", "winogender"])
    parser.add_argument("--model", type=str, default="gpt2-medium",
                       choices=["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl",
                               "gpt-neo-125m", "gpt-neo-1.3b", "gpt-neo-2.7b"],
                       help="Model to use for analysis")
    parser.add_argument("--output-dir", type=str, default="results/bias_vectors")
    parser.add_argument("--use-acdc", action="store_true", help="Use ACDC-formatted pair files")
    parser.add_argument("--target-layer", type=int, default=None, help="Target layer for steering (default: auto-detect)")
    parser.add_argument("--skip-extraction", action="store_true", help="Skip vector extraction")
    parser.add_argument("--skip-steering", action="store_true", help="Skip steering experiments")
    parser.add_argument("--skip-gradient-search", action="store_true", help="Skip gradient-based search")
    parser.add_argument("--n-opt-steps", type=int, default=50, help="Optimization steps for gradient search")
    parser.add_argument("--use-wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default="mech-interp-bias", help="W&B project name")
    parser.add_argument("--wandb-entity", type=str, default=None, help="W&B entity/username")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize W&B if requested
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=f"bias_vectors_{args.dataset}_{args.model}",
            config={
                "dataset": args.dataset,
                "model": args.model,
                "target_layer": args.target_layer,
                "n_opt_steps": args.n_opt_steps,
                "section": "4.3_bias_vectors"
            },
            tags=["bias_vectors", "steering", "gradient_search", args.dataset]
        )
        print("✅ Weights & Biases initialized!")
    
    print(f"\n{'='*70}")
    print("Section 4.3: Bias Vector Identification")
    print(f"{'='*70}")
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model}")
    print(f"Output directory: {output_dir}")
    print(f"W&B logging: {args.use_wandb}")
    print(f"Using ACDC pairs: {args.use_acdc}")
    print(f"{'='*70}\n")
    
    # Load model
    print("Loading model...")
    model = load_model(args.model)
    tokenizer = get_tokenizer(args.model)
    n_layers = model.cfg.n_layers
    
    # Prepare data
    print(f"\nPreparing {args.dataset} data...")
    biased_prompts, neutral_prompts, steering_prompts = prepare_bias_prompts(
        args.dataset,
        use_acdc=args.use_acdc
    )
    
    print(f"Biased prompts: {len(biased_prompts)}")
    print(f"Neutral prompts: {len(neutral_prompts)}")
    print(f"Steering prompts: {len(steering_prompts)}")
    
    # Part 1: Extract bias vectors from all layers
    if not args.skip_extraction:
        print(f"\n{'='*70}")
        print("Part 1: Mean Difference Vectors (with Enhancement)")
        print(f"{'='*70}\n")
        
        vectors_dict = extract_bias_vectors_all_layers(
            model=model,
            biased_prompts=biased_prompts,
            neutral_prompts=neutral_prompts,
            tokenizer=tokenizer,
            enhance=True
        )
        
        # Visualize
        print("\nVisualizing bias vectors...")
        from src.methods.bias_vectors.mean_difference import visualize_bias_vectors
        visualize_bias_vectors(
            vectors_dict,
            output_dir=output_dir,
            save_prefix=f"vectors_{args.dataset}"
        )
        
        # Save vectors
        vectors_file = output_dir / f"bias_vectors_{args.dataset}.pt"
        torch.save(vectors_dict, vectors_file)
        print(f"\nBias vectors saved to: {vectors_file}")
        
        # Determine best layer (highest enhanced norm)
        if args.target_layer is None:
            target_layer = max(vectors_dict.keys(), 
                             key=lambda l: vectors_dict[l].get('enhanced_norm', vectors_dict[l]['norm']))
            print(f"\nAuto-detected target layer: {target_layer}")
        else:
            target_layer = args.target_layer
        
        # Log to W&B
        if args.use_wandb:
            # Log per-layer vector properties
            for layer, data in vectors_dict.items():
                wandb.log({
                    f"vectors/layer_{layer}/norm": data['norm'],
                    "layer": layer
                })
                if 'enhanced_norm' in data:
                    wandb.log({
                        f"vectors/layer_{layer}/enhanced_norm": data['enhanced_norm'],
                        "layer": layer
                    })
                if 'analysis' in data:
                    analysis = data['analysis']
                    if 'pca_alignment' in analysis:
                        wandb.log({
                            f"vectors/layer_{layer}/pca_alignment": analysis['pca_alignment'],
                            "layer": layer
                        })
            
            # Log summary
            max_norm = max(vectors_dict[l]['norm'] for l in vectors_dict.keys())
            wandb.log({
                "vectors/target_layer": target_layer,
                "vectors/max_norm": max_norm
            })
            
            # Log visualizations
            norm_plot = output_dir / f"vectors_{args.dataset}_norms.png"
            if norm_plot.exists():
                wandb.log({"vectors/norms_plot": wandb.Image(str(norm_plot))})
            
            pca_plot = output_dir / f"vectors_{args.dataset}_pca_alignment.png"
            if pca_plot.exists():
                wandb.log({"vectors/pca_alignment_plot": wandb.Image(str(pca_plot))})
            
            print("✅ Bias vector results logged to W&B")
    else:
        # Load pre-extracted vectors
        vectors_file = output_dir / f"bias_vectors_{args.dataset}.pt"
        if vectors_file.exists():
            vectors_dict = torch.load(vectors_file)
            print(f"Loaded bias vectors from: {vectors_file}")
            target_layer = args.target_layer if args.target_layer is not None else n_layers // 2
        else:
            print("No pre-extracted vectors found. Please run without --skip-extraction first.")
            return
    
    # Extract bias vectors for steering
    bias_vectors = {}
    for layer, data in vectors_dict.items():
        bias_vectors[layer] = data.get('enhanced_vector', data['vector'])
    
    # Part 2: Steering Experiments
    if not args.skip_steering:
        print(f"\n{'='*70}")
        print(f"Part 2: Steering Experiments (Layer {target_layer})")
        print(f"{'='*70}\n")
        
        steering_results = batch_steering_experiment(
            model=model,
            prompts=steering_prompts,
            tokenizer=tokenizer,
            bias_vectors=bias_vectors,
            target_layer=target_layer,
            steering_coeffs=[-2.0, -1.0, 0.0, 1.0, 2.0]
        )
        
        # Visualize
        print("\nVisualizing steering results...")
        visualize_steering_results(
            steering_results,
            output_dir=output_dir / "steering",
            save_prefix=f"steering_{args.dataset}"
        )
        
        # Log to W&B
        if args.use_wandb:
            # Log steering effects
            for coeff, mean_bias in zip(steering_results['steering_coeffs'], 
                                        steering_results['mean_bias_scores']):
                wandb.log({
                    f"steering/coeff_{coeff}/mean_bias": mean_bias,
                    "steering_coefficient": coeff
                })
            
            # Log baseline and effects
            baseline_idx = steering_results['steering_coeffs'].index(0.0)
            wandb.log({
                "steering/baseline_bias": steering_results['mean_bias_scores'][baseline_idx],
                "steering/negative_effect": steering_results['mean_bias_scores'][0],
                "steering/positive_effect": steering_results['mean_bias_scores'][-1],
                "steering/target_layer": target_layer
            })
            
            # Log visualization
            steering_plot = output_dir / "steering" / f"steering_{args.dataset}_layer{target_layer}.png"
            if steering_plot.exists():
                wandb.log({"steering/effect_plot": wandb.Image(str(steering_plot))})
            
            print("✅ Steering results logged to W&B")
        
        # Generate example completions
        print("\n" + "="*70)
        print("Generating Example Completions with Steering")
        print("="*70 + "\n")
        
        example_prompts = [p["text"] for p in steering_prompts[:5]]
        examples = generate_steering_examples(
            model=model,
            prompts=example_prompts,
            tokenizer=tokenizer,
            bias_vector=bias_vectors[target_layer],
            layer_idx=target_layer,
            steering_coeffs=[-2.0, 0.0, 2.0],
            max_new_tokens=15,
            output_file=output_dir / "steering" / f"examples_{args.dataset}.json"
        )
    
    # Part 3: Gradient-Based Vector Search (Novel)
    if not args.skip_gradient_search:
        print(f"\n{'='*70}")
        print(f"Part 3: Gradient-Based Vector Search (Novel Contribution)")
        print(f"{'='*70}\n")
        
        # Get mean-difference vector for comparison
        mean_diff_vec = bias_vectors.get(target_layer)
        
        gradient_results = optimize_bias_direction(
            model=model,
            biased_prompts=biased_prompts[:200],  # Use 200 samples for better statistics
            neutral_prompts=neutral_prompts[:200],
            tokenizer=tokenizer,
            layer_idx=target_layer,
            mean_diff_vector=mean_diff_vec,
            n_steps=args.n_opt_steps,
            lr=0.01
        )
        
        # Visualize
        print("\nVisualizing gradient search results...")
        visualize_gradient_search(
            gradient_results,
            output_dir=output_dir / "gradient_search",
            save_prefix=f"gradient_{args.dataset}"
        )
        
        # Save optimized vectors
        opt_vectors_file = output_dir / "gradient_search" / f"optimized_vectors_{args.dataset}.pt"
        torch.save(gradient_results, opt_vectors_file)
        print(f"\nOptimized vectors saved to: {opt_vectors_file}")
        
        # Log to W&B
        if args.use_wandb:
            # Log optimization curves
            random_loss = gradient_results['random_init']['loss_history']
            for step, loss in enumerate(random_loss):
                wandb.log({
                    "gradient_search/random_init/loss": loss,
                    "gradient_search/random_init/step": step
                })
            
            # Log final results
            wandb.log({
                "gradient_search/random_init/final_loss": gradient_results['random_init']['final_loss'],
                "gradient_search/random_init/similarity_to_mean_diff": gradient_results['random_init']['similarity_to_mean_diff'],
                "gradient_search/target_layer": target_layer,
                "gradient_search/n_steps": args.n_opt_steps
            })
            
            # If mean-diff initialization exists
            if 'mean_diff_init' in gradient_results:
                mean_loss = gradient_results['mean_diff_init']['loss_history']
                for step, loss in enumerate(mean_loss):
                    wandb.log({
                        "gradient_search/mean_diff_init/loss": loss,
                        "gradient_search/mean_diff_init/step": step
                    })
                
                wandb.log({
                    "gradient_search/mean_diff_init/final_loss": gradient_results['mean_diff_init']['final_loss'],
                    "gradient_search/mean_diff_init/similarity_to_mean_diff": gradient_results['mean_diff_init']['similarity_to_mean_diff']
                })
            
            # Log visualization
            gradient_plot = output_dir / "gradient_search" / f"gradient_{args.dataset}_layer{target_layer}.png"
            if gradient_plot.exists():
                wandb.log({"gradient_search/optimization_curves": wandb.Image(str(gradient_plot))})
            
            print("✅ Gradient search results logged to W&B")
    
    print(f"\n{'='*70}")
    print("Bias vector analysis complete!")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*70}\n")
    
    # Finish W&B run
    if args.use_wandb:
        wandb.finish()
        print("✅ W&B run finished")


if __name__ == "__main__":
    main()

