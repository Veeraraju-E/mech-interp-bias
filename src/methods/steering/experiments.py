import argparse
import copy
import json
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import torch

from src.model_setup import get_tokenizer, load_model, setup_device
from .dataset import collect_activation_dataset
from .evaluation import evaluate_bias_reduction, evaluate_perplexity
from .feature_analysis import compute_latent_correlations, select_bias_latents
from .mmlu import DEFAULT_MMLU_SUBJECTS, evaluate_mmlu_accuracy, load_mmlu_samples
from .suppression import create_suppression_hook
from .train_sae import train_sae
from .visualization import plot_activation_profile, plot_latent_profile


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
    *,
    mmlu_subjects: Optional[Sequence[str]] = None,
    mmlu_max_questions: int = 25,
    skip_mmlu: bool = False,
    plot_activation_hist: bool = True,
    activation_hist_samples: int = 256,
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
    bias_mask = np.array(
        [entry.get("male_prob", 0.0) > entry.get("female_prob", 0.0) for entry in entries],
        dtype=bool,
    )

    results = []
    text_corpus = prompts[: min(50, len(prompts))]

    # Optional activation histogram for diagnostics
    histogram_stats = None
    if plot_activation_hist:
        biased_prompts = [
            entry["prompt"]
            for entry in entries
            if entry.get("male_prob", 0.0) > entry.get("female_prob", 0.0)
        ]
        neutral_prompts = [
            entry["prompt"]
            for entry in entries
            if entry.get("male_prob", 0.0) <= entry.get("female_prob", 0.0)
        ]
        if biased_prompts and neutral_prompts:
            hist_dir = output_dir / "visualizations"
            hist_path = hist_dir / f"activation_profile_layer{layer}.png"
            raw_profile = plot_activation_profile(
                model=model,
                tokenizer=tokenizer,
                biased_prompts=biased_prompts,
                neutral_prompts=neutral_prompts,
                layer=layer,
                max_samples=activation_hist_samples,
                output_path=hist_path,
            )
            profile_dir = output_dir / "activation_profiles"
            profile_dir.mkdir(parents=True, exist_ok=True)
            profile_data = {
                "biased_means": raw_profile["biased_means"].tolist(),
                "neutral_means": raw_profile["neutral_means"].tolist(),
                "num_neurons": raw_profile["num_neurons"],
                "num_biased_prompts": raw_profile["num_biased_prompts"],
                "num_neutral_prompts": raw_profile["num_neutral_prompts"],
            }
            histogram_stats = profile_data
            profile_path = profile_dir / f"layer{layer}.json"
            with open(profile_path, "w") as f:
                json.dump(profile_data, f, indent=2)
            print(f"Saved activation profile plot to {hist_path}")
        else:
            print("Skipping activation histogram (insufficient biased/neutral prompts).")

    # Pre-load MMLU samples to share across SAE settings
    mmlu_samples = []
    baseline_mmlu_metrics = None
    effective_subjects = None
    if not skip_mmlu:
        effective_subjects = (
            list(mmlu_subjects) if mmlu_subjects else list(DEFAULT_MMLU_SUBJECTS)
        )
        print(
            f"Loading MMLU subjects {effective_subjects} "
            f"(<= {mmlu_max_questions} questions per subject)"
        )
        mmlu_samples = load_mmlu_samples(
            effective_subjects,
            max_questions_per_subject=mmlu_max_questions,
        )
        if mmlu_samples:
            baseline_mmlu_metrics = evaluate_mmlu_accuracy(
                model=model,
                tokenizer=tokenizer,
                samples=mmlu_samples,
            )
            print(
                f"Baseline MMLU accuracy: "
                f"{baseline_mmlu_metrics['overall_accuracy']:.3f} "
                f"over {baseline_mmlu_metrics['num_questions']} questions"
            )
        else:
            print("Warning: No MMLU samples were loaded; skipping MMLU evaluation.")

    for k_sparse in k_sparse_values:
        print(f"\n{'='*60}\nTraining SAE with k_sparse={k_sparse}\n{'='*60}")
        sae, _ = train_sae(
            activations=activations,
            d_model=model.cfg.d_model,
            d_latent=d_latent or model.cfg.d_model * 8,
            k_sparse=k_sparse,
            device=device,
            epochs=1000
        )

        correlations = compute_latent_correlations(sae, activations, np.array(gld_scores))
        suppressed_latents = select_bias_latents(correlations, min_abs_corr=0.05, top_k=top_latents)
        if not suppressed_latents:
            print("No latents exceeded correlation threshold; skipping suppression.")
            continue

        bias_metrics = evaluate_bias_reduction(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            sae=sae,
            suppressed_latents=suppressed_latents,
            layer=layer,
            scale=suppression_scale,
        )

        perplexity_metrics = evaluate_perplexity(
            model=model,
            tokenizer=tokenizer,
            text_corpus=text_corpus,
            sae=sae,
            suppressed_latents=suppressed_latents,
            layer=layer,
            scale=suppression_scale,
        )

        # Compute SAE latent activations for plotting
        latent_profile_info = None
        with torch.no_grad():
            latent_batches = []
            batch_size = 512
            for start in range(0, activations_tensor.shape[0], batch_size):
                batch = activations_tensor[start : start + batch_size]
                lat_batch = sae.encode(batch, apply_sparsity=True)
                latent_batches.append(lat_batch.cpu())
            latents = torch.cat(latent_batches, dim=0).numpy()

        if latents.shape[0] == bias_mask.shape[0]:
            biased_latents = latents[bias_mask]
            neutral_latents = latents[~bias_mask]
            if biased_latents.size and neutral_latents.size:
                latent_plot_dir = output_dir / "latent_profiles"
                latent_plot_dir.mkdir(parents=True, exist_ok=True)
                latent_plot_path = latent_plot_dir / f"layer{layer}_k{k_sparse}.png"
                latent_stats = plot_latent_profile(
                    biased_latents,
                    neutral_latents,
                    output_path=latent_plot_path,
                    title=f"Layer {layer} SAE Latent Means (k={k_sparse})",
                )
                latent_profile_info = {
                    "plot": str(latent_plot_path),
                    "biased_means": latent_stats["biased_means"].tolist(),
                    "neutral_means": latent_stats["neutral_means"].tolist(),
                    "mean_difference": latent_stats["mean_difference"].tolist(),
                    "num_latents": latent_stats["num_latents"],
                    "num_biased_samples": latent_stats["num_biased_samples"],
                    "num_neutral_samples": latent_stats["num_neutral_samples"],
                }
        else:
            print("Warning: latent activations size mismatch; skipping latent profile plot.")

        mmlu_result = None
        if mmlu_samples and baseline_mmlu_metrics:
            hook_name = (
                "blocks.0.hook_resid_pre"
                if layer == 0
                else f"blocks.{layer-1}.hook_resid_post"
            )
            steering_hooks = [
                (
                    hook_name,
                    create_suppression_hook(
                        sae=sae,
                        suppressed_latents=suppressed_latents,
                        scale=suppression_scale,
                    ),
                )
            ]
            steered_metrics = evaluate_mmlu_accuracy(
                model=model,
                tokenizer=tokenizer,
                samples=mmlu_samples,
                steering_hooks=steering_hooks,
            )
            mmlu_result = {
                "baseline": copy.deepcopy(baseline_mmlu_metrics),
                "steered": steered_metrics,
            }
            print(
                "MMLU overall accuracy: "
                f"{baseline_mmlu_metrics['overall_accuracy']:.3f} (baseline) → "
                f"{steered_metrics['overall_accuracy']:.3f} (steered)"
            )

        result = {
            "k_sparse": k_sparse,
            "suppressed_latents": suppressed_latents,
            "bias_metrics": bias_metrics,
            "perplexity_metrics": perplexity_metrics,
        }
        if mmlu_result is not None:
            result["mmlu_accuracy"] = mmlu_result
        if latent_profile_info is not None:
            result["latent_profile"] = latent_profile_info
        results.append(result)

        save_path = output_dir / f"sae_bias_suppression_k{k_sparse}.json"
        with open(save_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Saved result to {save_path}")

    summary = {
        "model": model_name,
        "dataset": dataset_name,
        "layer": layer,
        "d_latent": d_latent or model.cfg.d_model * 8,
        "suppression_scale": suppression_scale,
        "top_latents": top_latents,
        "mmlu_subjects": effective_subjects if mmlu_samples else [],
        "baseline_mmlu_accuracy": baseline_mmlu_metrics,
        "activation_histogram": histogram_stats,
        "results": results,
    }
    summary_path = output_dir / "sae_bias_suppression_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SAE steering experiment (Hegde 2024)")
    parser.add_argument("--model", type=str, default="gpt2-medium")
    parser.add_argument("--dataset", type=str, default="stereoset_gender", choices=["stereoset_gender", "winogender"])
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--d-latent", type=int, default=None)
    parser.add_argument("--k-sparse", type=int, nargs="+", default=[64])
    parser.add_argument("--n-examples", type=int, default=2000)
    parser.add_argument("--suppression-scale", type=float, default=0.0)
    parser.add_argument("--top-latents", type=int, default=32)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--mmlu-subjects", type=str, default=",".join(DEFAULT_MMLU_SUBJECTS))
    parser.add_argument("--mmlu-max-questions", type=int, default=25)
    parser.add_argument("--no-mmlu", action="store_true", help="Skip MMLU evaluation for faster debugging.")
    parser.add_argument("--activation-hist-samples", type=int, default=256)
    parser.add_argument("--no-activation-hist", action="store_true", help="Disable the activation histogram visualization.")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir) if args.output_dir else Path("results") / args.model / "sae_bias_suppression"
    subjects = [
        subject.strip()
        for subject in (args.mmlu_subjects.split(",") if args.mmlu_subjects else [])
        if subject.strip()
    ]
    run_experiment(
        model_name=args.model,
        dataset_name=args.dataset,
        layer=args.layer,
        d_latent=args.d_latent,
        k_sparse_values=args.k_sparse,
        n_examples=args.n_examples,
        suppression_scale=args.suppression_scale,
        top_latents=args.top_latents,
        output_dir=output_dir,
        mmlu_subjects=subjects,
        mmlu_max_questions=args.mmlu_max_questions,
        skip_mmlu=args.no_mmlu,
        plot_activation_hist=not args.no_activation_hist,
        activation_hist_samples=args.activation_hist_samples,
    )


if __name__ == "__main__":
    main()