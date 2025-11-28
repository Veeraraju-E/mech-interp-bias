"""SAE steering package aligned with Hegde (2024)."""

from .dataset import collect_activation_dataset, build_gender_prompts
from .train_sae import train_sae
from .feature_analysis import compute_latent_correlations, select_bias_latents
from .suppression import create_suppression_hook, apply_suppression
from .evaluation import evaluate_bias_reduction, evaluate_perplexity
from .experiments import run_experiment
from .mmlu import DEFAULT_MMLU_SUBJECTS, evaluate_mmlu_accuracy, load_mmlu_samples
from .visualization import plot_activation_profile, plot_latent_profile

__all__ = [
    "collect_activation_dataset",
    "build_gender_prompts",
    "train_sae",
    "compute_latent_correlations",
    "select_bias_latents",
    "create_suppression_hook",
    "apply_suppression",
    "evaluate_bias_reduction",
    "evaluate_perplexity",
    "evaluate_mmlu_accuracy",
    "load_mmlu_samples",
    "DEFAULT_MMLU_SUBJECTS",
    "plot_activation_profile",
    "plot_latent_profile",
    "run_experiment",
]

