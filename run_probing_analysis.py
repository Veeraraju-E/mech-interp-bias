"""

This script demonstrates:
1. Linear probes trained on hidden-layer activations
2. Logit Lens analysis for bias emergence
3. Identification of layers where bias is encoded

Usage:
    python run_probing_analysis.py --dataset stereoset --model gpt2-medium
"""

import argparse
import json
from pathlib import Path
import torch
import wandb

from src.model_setup import load_model, get_tokenizer
from src.data_loader import (
    load_stereoset, load_winogender, get_stereoset_pairs, build_winogender_pairs,
    load_acdc_pairs, get_acdc_stereoset_pairs, get_acdc_winogender_pairs
)
from src.methods.probing import (
    train_layerwise_probes,
    plot_probe_results,
    analyze_bias_emergence,
    visualize_logit_lens,
    evaluate_layer_ablation,
    find_biased_heads,
    visualize_intervention_results,
    plot_activation_clustering,
    plot_token_probability_heatmap,
    plot_attention_flow,
    load_attribution_results,
    evaluate_attribution_guided_ablation
)


def prepare_bias_prompts(dataset_name: str, data_dir: Path = Path("data"), use_acdc: bool = False):
    """
    Prepare biased and neutral prompts from datasets.
    
    Args:
        dataset_name: "stereoset", "stereoset_gender", "stereoset_race", or "winogender"
        data_dir: Directory containing data files
        use_acdc: If True, use ACDC-formatted pairs
    
    Returns:
        biased_prompts, neutral_prompts, logit_lens_prompts
    """
    if use_acdc:
        # Use new ACDC pair format
        if dataset_name.startswith("stereoset"):
            if dataset_name == "stereoset":
                # Default to gender bias
                pairs = get_acdc_stereoset_pairs("gender", data_dir)
            else:
                # Extract bias type: stereoset_gender or stereoset_race
                bias_type = dataset_name.split("_")[1]
                pairs = get_acdc_stereoset_pairs(bias_type, data_dir)
            
            biased_prompts = []
            neutral_prompts = []
            logit_lens_prompts = []
            
            for pair in pairs[:100]:  # Limit for faster testing
                clean = pair.get("clean", {})
                corrupted = pair.get("corrupted", {})
                metadata = clean.get("metadata", {})
                
                # Determine which is biased based on label
                label = metadata.get("label", "stereotype")
                if label == "stereotype":
                    biased_prompts.append(clean["sentence"])
                    neutral_prompts.append(corrupted["sentence"])
                else:  # antistereotype
                    biased_prompts.append(corrupted["sentence"])
                    neutral_prompts.append(clean["sentence"])
                
                # For logit lens: use context
                context = clean.get("context", "")
                if context:
                    logit_lens_prompts.append({
                        "text": context,
                        "biased_tokens": ["he", "him", "his", "man", "male"],
                        "neutral_tokens": ["she", "her", "hers", "woman", "female"],
                        "stereo_word": metadata.get("stereotype_word", ""),
                        "antistereo_word": metadata.get("antistereotype_word", "")
                    })
        
        elif dataset_name == "winogender":
            pairs = get_acdc_winogender_pairs(data_dir)
            
            biased_prompts = []
            neutral_prompts = []
            logit_lens_prompts = []
            
            for pair in pairs[:100]:
                clean = pair.get("clean", {})
                corrupted = pair.get("corrupted", {})
                metadata = clean.get("metadata", {})
                
                # For winogender, clean is female, corrupted is male
                biased_prompts.append(corrupted["sentence"])  # male version
                neutral_prompts.append(clean["sentence"])  # female version
                
                # For logit lens
                context = clean.get("context", "")
                if context:
                    logit_lens_prompts.append({
                        "text": context,
                        "biased_tokens": ["he", "him", "his"],
                        "neutral_tokens": ["she", "her", "hers"],
                        "profession": metadata.get("profession", "")
                    })
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
    
    else:
        # Use original format (legacy support)
        if dataset_name == "stereoset":
            examples = load_stereoset(data_dir)
            pairs = get_stereoset_pairs(examples)
            
            # Filter for gender bias only
            gender_pairs = [p for p in pairs if p.get("bias_type") == "gender"]
            
            # Biased prompts: context that leads to stereotype
            biased_prompts = []
            neutral_prompts = []
            
            for pair in gender_pairs[:100]:  # Limit for faster testing
                prefix = pair.get("context_prefix", pair.get("context", ""))
                if prefix:
                    # Use stereotype completion as biased
                    biased_prompts.append(pair["stereotype_sentence"])
                    # Use antistereotype as neutral
                    neutral_prompts.append(pair["antistereotype_sentence"])
            
            # For logit lens: prepare prompts with token expectations
            logit_lens_prompts = []
            for pair in gender_pairs[:50]:
                prefix = pair.get("context_prefix", "")
                if prefix:
                    logit_lens_prompts.append({
                        "text": prefix,
                        "biased_tokens": ["he", "him", "his", "man", "male"],
                        "neutral_tokens": ["she", "her", "hers", "woman", "female"]
                    })
        
        elif dataset_name == "winogender":
            examples = load_winogender(data_dir)
            pairs = build_winogender_pairs(examples)
            
            biased_prompts = []
            neutral_prompts = []
            
            for pair in pairs[:100]:
                prompt = pair.get("prompt", "")
                if prompt:
                    # Male version as "biased" (just for demonstration)
                    biased_prompts.append(pair.get("male_sentence", ""))
                    # Female version as "neutral"
                    neutral_prompts.append(pair.get("female_sentence", ""))
            
            # For logit lens
            logit_lens_prompts = []
            for pair in pairs[:50]:
                prompt = pair.get("prompt", "")
                if prompt:
                    logit_lens_prompts.append({
                        "text": prompt,
                        "biased_tokens": ["he", "him", "his"],
                        "neutral_tokens": ["she", "her", "hers"]
                    })
        
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return biased_prompts, neutral_prompts, logit_lens_prompts


def main():
    parser = argparse.ArgumentParser(description="Run probing and layer analysis")
    parser.add_argument("--dataset", type=str, default="stereoset", 
                       choices=["stereoset", "stereoset_gender", "stereoset_race", "winogender"])
    parser.add_argument("--model", type=str, default="gpt2-medium",
                       choices=["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl", 
                               "gpt-neo-125m", "gpt-neo-1.3b", "gpt-neo-2.7b"],
                       help="Model to use for analysis")
    parser.add_argument("--output-dir", type=str, default="results/probing")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--target-layer", type=int, default=None, help="Specific layer to analyze (e.g., 3 for layer 3)")
    parser.add_argument("--use-acdc", action="store_true", help="Use ACDC-formatted pair files")
    parser.add_argument("--skip-probes", action="store_true", help="Skip probe training")
    parser.add_argument("--skip-logit-lens", action="store_true", help="Skip logit lens analysis")
    parser.add_argument("--skip-interventions", action="store_true", help="Skip probe-guided interventions")
    parser.add_argument("--use-wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default="mech-interp-bias", help="W&B project name")
    parser.add_argument("--wandb-entity", type=str, default=None, help="W&B entity/username")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize W&B if requested
    if args.use_wandb:
        config = {
            "dataset": args.dataset,
            "model": args.model,
            "test_size": args.test_size,
            "section": "4.2_probing"
        }
        if args.target_layer is not None:
            config["target_layer"] = args.target_layer
        
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=f"probing_{args.dataset}_{args.model}" + (f"_layer{args.target_layer}" if args.target_layer is not None else ""),
            config=config,
            tags=["probing", "layer_analysis", args.dataset] + ([f"layer_{args.target_layer}"] if args.target_layer is not None else [])
        )
        print("✅ Weights & Biases initialized!")
    
    print(f"\n{'='*70}")
    print("Section 4.2: Probing and Layer Analysis")
    print(f"{'='*70}")
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model}")
    print(f"Output directory: {output_dir}")
    print(f"W&B logging: {args.use_wandb}")
    print(f"Using ACDC pairs: {args.use_acdc}")
    if args.target_layer is not None:
        print(f"🎯 Target layer for focused analysis: {args.target_layer}")
    print(f"{'='*70}\n")
    
    # Load model
    print("Loading model...")
    model = load_model(args.model)
    tokenizer = get_tokenizer(args.model)
    
    # Prepare data
    print(f"\nPreparing {args.dataset} data...")
    biased_prompts, neutral_prompts, logit_lens_prompts = prepare_bias_prompts(
        args.dataset, 
        use_acdc=args.use_acdc
    )
    
    print(f"Biased prompts: {len(biased_prompts)}")
    print(f"Neutral prompts: {len(neutral_prompts)}")
    print(f"Logit lens prompts: {len(logit_lens_prompts)}")
    
    # Part 1: Linear Probes
    if not args.skip_probes:
        print(f"\n{'='*70}")
        print("Part 1: Training Linear Probes")
        print(f"{'='*70}\n")
        
        probe_results = train_layerwise_probes(
            model=model,
            biased_prompts=biased_prompts,
            neutral_prompts=neutral_prompts,
            tokenizer=tokenizer,
            test_size=args.test_size,
            hook_name="hook_resid_post"
        )
        
        # Visualize results
        print("\nVisualizing probe results...")
        plot_probe_results(
            probe_results,
            output_dir=output_dir,
            save_prefix=f"probe_{args.dataset}"
        )
        
        # Find best layer
        best_layer = max(probe_results.keys(), key=lambda l: probe_results[l][1]['auc'])
        best_auc = probe_results[best_layer][1]['auc']
        print(f"\nBest layer for bias detection: {best_layer} (AUC={best_auc:.4f})")
        
        # Log to W&B
        if args.use_wandb:
            # Log per-layer metrics
            for layer, (probe, metrics) in probe_results.items():
                wandb.log({
                    f"probe/layer_{layer}/accuracy": metrics['accuracy'],
                    f"probe/layer_{layer}/auc": metrics['auc'],
                    f"probe/layer_{layer}/f1": metrics['f1'],
                    "layer": layer
                })
            
            # Log summary metrics
            wandb.log({
                "probe/best_layer": best_layer,
                "probe/best_auc": best_auc,
                "probe/max_accuracy": max(probe_results[l][1]['accuracy'] for l in probe_results.keys()),
                "probe/max_f1": max(probe_results[l][1]['f1'] for l in probe_results.keys())
            })
            
            # Log visualization
            probe_plot = output_dir / f"probe_{args.dataset}_layerwise.png"
            if probe_plot.exists():
                wandb.log({"probe/layerwise_plot": wandb.Image(str(probe_plot))})
                print("✅ Probe results logged to W&B")
        
        # Enhanced visualization: Activation clustering
        if probe_results:
            print("\n" + "="*70)
            print("Enhanced Visualization: Activation Clustering")
            print("="*70)
            
            # Get best layer by AUC, or use target layer if specified
            if args.target_layer is not None:
                best_probe_layer = args.target_layer
                print(f"\n🎯 Using specified target layer: {best_probe_layer}")
            else:
                best_probe_layer = max(probe_results.keys(), 
                                     key=lambda l: probe_results[l][1]['auc'])
                print(f"\nGenerating activation clustering for best probe layer: {best_probe_layer}")
            
            try:
                plot_activation_clustering(
                    model=model,
                    biased_prompts=biased_prompts[:50],  # Sample for speed
                    neutral_prompts=neutral_prompts[:50],
                    tokenizer=tokenizer,
                    layer_idx=best_probe_layer,
                    method="tsne",
                    output_dir=output_dir,
                    save_prefix=f"activation_cluster_{args.dataset}"
                )
                
                # Also generate PCA version
                plot_activation_clustering(
                    model=model,
                    biased_prompts=biased_prompts[:50],
                    neutral_prompts=neutral_prompts[:50],
                    tokenizer=tokenizer,
                    layer_idx=best_probe_layer,
                    method="pca",
                    output_dir=output_dir,
                    save_prefix=f"activation_cluster_{args.dataset}"
                )
                
                # Log to W&B
                if args.use_wandb:
                    tsne_plot = output_dir / f"activation_cluster_{args.dataset}_layer{best_probe_layer}_tsne.png"
                    pca_plot = output_dir / f"activation_cluster_{args.dataset}_layer{best_probe_layer}_pca.png"
                    if tsne_plot.exists():
                        wandb.log({"probe/activation_clustering_tsne": wandb.Image(str(tsne_plot))})
                    if pca_plot.exists():
                        wandb.log({"probe/activation_clustering_pca": wandb.Image(str(pca_plot))})
                
                print("✅ Activation clustering complete")
            except Exception as e:
                print(f"⚠️  Warning: Activation clustering failed: {e}")
                print("   Continuing with other analyses...")
    
    # Part 2: Logit Lens Analysis
    if not args.skip_logit_lens:
        print(f"\n{'='*70}")
        print("Part 2: Logit Lens Analysis")
        print(f"{'='*70}\n")
        
        logit_lens_results = analyze_bias_emergence(
            model=model,
            prompts=logit_lens_prompts,
            tokenizer=tokenizer,
            dataset_type=args.dataset
        )
        
        # Visualize results
        print("\nVisualizing logit lens results...")
        visualize_logit_lens(
            logit_lens_results,
            output_dir=output_dir,
            save_prefix=f"logit_lens_{args.dataset}"
        )
        
        print(f"\nBias emergence layer: {logit_lens_results['bias_emergence_layer']}")
        
        # Save best layer for interventions
        best_layer = logit_lens_results['bias_emergence_layer']
        
        # Log to W&B
        if args.use_wandb:
            # Log per-layer bias scores
            for layer in range(logit_lens_results['n_layers']):
                if layer in logit_lens_results['layerwise_mean_bias']:
                    wandb.log({
                        f"logit_lens/layer_{layer}/mean_bias": logit_lens_results['layerwise_mean_bias'][layer],
                        f"logit_lens/layer_{layer}/std_bias": logit_lens_results['layerwise_std_bias'][layer],
                        "layer": layer
                    })
            
            # Log summary
            wandb.log({
                "logit_lens/bias_emergence_layer": logit_lens_results['bias_emergence_layer'],
                "logit_lens/max_mean_bias": max(logit_lens_results['layerwise_mean_bias'].values())
            })
            
            # Log visualization
            logit_plot = output_dir / f"logit_lens_{args.dataset}.png"
            if logit_plot.exists():
                wandb.log({"logit_lens/emergence_plot": wandb.Image(str(logit_plot))})
                print("✅ Logit lens results logged to W&B")
        
        # Enhanced visualization: Token probability heatmap
        if not args.skip_logit_lens and logit_lens_prompts:
            print("\n" + "="*70)
            print("Enhanced Visualization: Token Probability Heatmap")
            print("="*70)
            
            # Define target tokens based on dataset
            if args.dataset == "stereoset" or args.dataset == "winogender":
                target_tokens = ["he", "she", "his", "her", "him"]
            else:
                target_tokens = ["he", "she"]
            
            print(f"\nTracking token probabilities for: {target_tokens}")
            try:
                # Use a sample of prompts for speed
                # Extract text from dictionaries if needed
                sample_prompts = logit_lens_prompts[:30]
                if sample_prompts and isinstance(sample_prompts[0], dict):
                    sample_prompts = [p.get("text", str(p)) for p in sample_prompts]
                
                plot_token_probability_heatmap(
                    model=model,
                    prompts=sample_prompts,
                    tokenizer=tokenizer,
                    target_tokens=target_tokens,
                    output_dir=output_dir,
                    save_prefix=f"token_probability_{args.dataset}"
                )
                
                # Log to W&B
                if args.use_wandb:
                    heatmap_plot = output_dir / f"token_probability_{args.dataset}_heatmap.png"
                    evolution_plot = output_dir / f"token_probability_{args.dataset}_evolution.png"
                    if heatmap_plot.exists():
                        wandb.log({"logit_lens/token_probability_heatmap": wandb.Image(str(heatmap_plot))})
                    if evolution_plot.exists():
                        wandb.log({"logit_lens/token_probability_evolution": wandb.Image(str(evolution_plot))})
                
                print("✅ Token probability heatmap complete")
            except Exception as e:
                print(f"⚠️  Warning: Token probability heatmap failed: {e}")
                print("   Continuing with other analyses...")
    else:
        best_layer = None
    
    # Part 3: Probe-Guided Interventions
    if not args.skip_interventions:
        print(f"\n{'='*70}")
        print("Part 3: Probe-Guided Interventions (Section 4.2.3)")
        print(f"{'='*70}\n")
        
        # Determine layers to test based on target layer or probe results
        if args.target_layer is not None:
            # Focus on target layer and neighbors
            top_layers = [l for l in [args.target_layer-1, args.target_layer, args.target_layer+1] 
                         if 0 <= l < model.cfg.n_layers]
            print(f"🎯 Using target layer {args.target_layer} and neighbors: {top_layers}")
        elif not args.skip_probes and probe_results:
            # Get top 5 layers by probe AUC
            top_layers = sorted(probe_results.keys(), 
                              key=lambda l: probe_results[l][1]['auc'], 
                              reverse=True)[:5]
        elif best_layer is not None:
            # Use bias emergence layer and neighbors
            top_layers = [l for l in [best_layer-1, best_layer, best_layer+1] 
                         if 0 <= l < model.cfg.n_layers]
        else:
            # Default: test late layers
            top_layers = list(range(model.cfg.n_layers - 5, model.cfg.n_layers))
        
        print(f"Testing layer ablation on layers: {top_layers}")
        
        # Prepare general prompts for side-effect measurement
        general_prompts = [
            "The quick brown fox jumps over the lazy dog.",
            "In recent years, artificial intelligence has made significant progress.",
            "The study of language and linguistics is fascinating.",
            "Climate change is one of the most pressing issues of our time.",
            "Technology continues to evolve at an unprecedented pace."
        ]
        
        # Evaluate layer ablation
        intervention_results = evaluate_layer_ablation(
            model=model,
            bias_prompts=logit_lens_prompts if not args.skip_logit_lens else [],
            general_prompts=general_prompts,
            tokenizer=tokenizer,
            layers_to_test=top_layers,
            dataset_type=args.dataset
        )
        
        # Visualize
        print("\nVisualizing intervention results...")
        visualize_intervention_results(
            intervention_results,
            output_dir=output_dir / "interventions",
            save_prefix=f"interventions_{args.dataset}"
        )
        
        # Attribution-Guided Targeted Ablation (using Section 4.1 results)
        print(f"\n{'='*70}")
        print("Attribution-Guided Targeted Ablation (Section 4.1 → 4.2.3)")
        print(f"{'='*70}\n")
        
        attribution_dir = Path("results/attribution_patching")
        if attribution_dir.exists():
            print("Loading attribution patching results to guide targeted ablations...")
            try:
                attribution_results = load_attribution_results(
                    attribution_dir=attribution_dir,
                    dataset=args.dataset
                )
                
                if attribution_results:
                    print("\nEvaluating attribution-guided ablations...")
                    attr_ablation_results = evaluate_attribution_guided_ablation(
                        model=model,
                        bias_prompts=logit_lens_prompts if not args.skip_logit_lens else [],
                        general_prompts=general_prompts,
                        tokenizer=tokenizer,
                        attribution_results=attribution_results,
                        dataset_type=args.dataset,
                        top_k_heads=10,
                        top_k_mlps=5
                    )
                    
                    # Save results
                    attr_output_dir = output_dir / "interventions"
                    attr_output_dir.mkdir(parents=True, exist_ok=True)
                    attr_results_file = attr_output_dir / f"attribution_guided_{args.dataset}_results.json"
                    with open(attr_results_file, "w") as f:
                        json.dump(attr_ablation_results, f, indent=2)
                    
                    print(f"\n✅ Attribution-guided ablation results saved to {attr_results_file}")
                    
                    # Log to W&B
                    if args.use_wandb:
                        if "head_ablation" in attr_ablation_results:
                            h = attr_ablation_results["head_ablation"]
                            wandb.log({
                                "attribution_guided/head_ablation/bias_reduction": h.get("bias_reduction", 0),
                                "attribution_guided/head_ablation/perplexity_increase": h.get("perplexity_increase", 0),
                                "attribution_guided/head_ablation/effectiveness": h.get("effectiveness", 0)
                            })
                        
                        if "mlp_ablation" in attr_ablation_results:
                            m = attr_ablation_results["mlp_ablation"]
                            wandb.log({
                                "attribution_guided/mlp_ablation/bias_reduction": m.get("bias_reduction", 0),
                                "attribution_guided/mlp_ablation/perplexity_increase": m.get("perplexity_increase", 0),
                                "attribution_guided/mlp_ablation/effectiveness": m.get("effectiveness", 0)
                            })
                        
                        if "combined_ablation" in attr_ablation_results:
                            c = attr_ablation_results["combined_ablation"]
                            wandb.log({
                                "attribution_guided/combined_ablation/bias_reduction": c.get("bias_reduction", 0),
                                "attribution_guided/combined_ablation/perplexity_increase": c.get("perplexity_increase", 0),
                                "attribution_guided/combined_ablation/effectiveness": c.get("effectiveness", 0)
                            })
                else:
                    print("⚠️  No attribution results found. Skipping attribution-guided ablation.")
            except Exception as e:
                print(f"⚠️  Warning: Attribution-guided ablation failed: {e}")
                print("   Continuing with other analyses...")
        else:
            print(f"⚠️  Attribution patching directory not found: {attribution_dir}")
            print("   Run Section 4.1 (Activation Patching) first to generate attribution results.")
            print("   Skipping attribution-guided ablation...")
        
        # Find biased heads
        print("\nFinding biased attention heads...")
        biased_heads = find_biased_heads(
            model=model,
            bias_prompts=logit_lens_prompts if not args.skip_logit_lens else [],
            tokenizer=tokenizer,
            target_layers=top_layers[:3],  # Top 3 layers only for speed
            dataset_type=args.dataset,
            top_k=10
        )
        
        # Save biased heads
        heads_file = output_dir / "interventions" / f"biased_heads_{args.dataset}.json"
        with open(heads_file, "w") as f:
            json.dump([{"layer": l, "head": h, "contribution": c} 
                      for l, h, c in biased_heads], f, indent=2)
        
        # Enhanced visualization: Attention flow diagram
        if biased_heads:
            print("\n" + "="*70)
            print("Enhanced Visualization: Attention Flow Diagram")
            print("="*70)
            
            # Get example prompt
            example_prompt = logit_lens_prompts[0] if logit_lens_prompts else biased_prompts[0] if biased_prompts else "The doctor said that"
            
            # Extract text from dictionary if needed
            if isinstance(example_prompt, dict):
                example_prompt = example_prompt.get("text", str(example_prompt))
            
            print(f"\nVisualizing attention for top biased heads...")
            print(f"Example prompt: {example_prompt}")
            
            try:
                # Visualize top 5 biased heads
                top_heads = biased_heads[:5]
                # Extract (layer, head) tuples from (layer, head, contribution)
                top_heads_tuples = [(l, h) for l, h, c in top_heads]
                
                plot_attention_flow(
                    model=model,
                    prompt=example_prompt,
                    tokenizer=tokenizer,
                    biased_heads=top_heads_tuples,
                    output_dir=output_dir / "interventions",
                    save_prefix=f"attention_flow_{args.dataset}"
                )
                
                # Log to W&B
                if args.use_wandb:
                    attn_plot = output_dir / "interventions" / f"attention_flow_{args.dataset}.png"
                    attn_stereo_plot = output_dir / "interventions" / f"attention_flow_{args.dataset}_stereotypes.png"
                    if attn_plot.exists():
                        wandb.log({"interventions/attention_flow": wandb.Image(str(attn_plot))})
                    if attn_stereo_plot.exists():
                        wandb.log({"interventions/attention_stereotypes": wandb.Image(str(attn_stereo_plot))})
                
                print("✅ Attention flow diagram complete")
            except Exception as e:
                print(f"⚠️  Warning: Attention flow diagram failed: {e}")
                print("   Continuing with other analyses...")
        
        # Log to W&B
        if args.use_wandb:
            # Log intervention metrics
            for layer, result in intervention_results["layer_results"].items():
                wandb.log({
                    f"interventions/layer_{layer}/bias_reduction": result["bias_reduction"],
                    f"interventions/layer_{layer}/perplexity_increase": result["perplexity_increase"],
                    f"interventions/layer_{layer}/effectiveness": result["effectiveness"],
                    "layer": layer
                })
            
            # Log summary
            best_layer_int = max(intervention_results["layer_results"].keys(),
                                key=lambda l: intervention_results["layer_results"][l]["effectiveness"])
            wandb.log({
                "interventions/best_layer": best_layer_int,
                "interventions/best_effectiveness": intervention_results["layer_results"][best_layer_int]["effectiveness"],
                "interventions/baseline_bias": intervention_results["baseline_bias"],
                "interventions/baseline_perplexity": intervention_results["baseline_perplexity"]
            })
            
            # Log biased heads
            for i, (layer, head, contrib) in enumerate(biased_heads[:5]):
                wandb.log({
                    f"interventions/top_head_{i}/layer": layer,
                    f"interventions/top_head_{i}/head": head,
                    f"interventions/top_head_{i}/contribution": contrib
                })
            
            # Log visualization
            intervention_plot = output_dir / "interventions" / f"interventions_{args.dataset}_layer_ablation.png"
            if intervention_plot.exists():
                wandb.log({"interventions/ablation_plot": wandb.Image(str(intervention_plot))})
            
            print("✅ Intervention results logged to W&B")
    
    print(f"\n{'='*70}")
    print("Analysis complete!")
    print(f"Results saved to: {output_dir}")
    if not args.skip_interventions:
        print(f"Interventions saved to: {output_dir}/interventions")
    print(f"{'='*70}\n")
    
    # Finish W&B run
    if args.use_wandb:
        wandb.finish()
        print("✅ W&B run finished")


if __name__ == "__main__":
    main()

