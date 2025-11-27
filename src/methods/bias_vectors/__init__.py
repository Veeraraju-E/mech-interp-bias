"""Bias vector identification and steering methods."""

from .mean_difference import (
    compute_mean_difference_vector,
    enhance_bias_vector,
    extract_bias_vectors_all_layers,
    visualize_bias_vectors
)

from .steering import (
    steer_generation,
    evaluate_steering_effect,
    batch_steering_experiment,
    visualize_steering_results,
    generate_steering_examples
)

from .gradient_search import (
    gradient_based_vector_search,
    optimize_bias_direction,
    visualize_gradient_search
)

# Projection-based steering (Section 4.1 & 4.2)
from .projection_steering import (
    orthogonal_projection,
    projection_steering_hook,
    evaluate_projection_steering,
    batch_projection_steering_experiment,
    visualize_projection_steering_results,
    generate_projection_steering_examples
)

from .sae_steering import (
    SparseAutoencoder,
    train_sae_on_activations,
    extract_bias_direction_from_sae,
    extract_sae_bias_vector,
    batch_sae_steering_experiment
)

__all__ = [
    "compute_mean_difference_vector",
    "enhance_bias_vector",
    "extract_bias_vectors_all_layers",
    "visualize_bias_vectors",
    "steer_generation",
    "evaluate_steering_effect",
    "batch_steering_experiment",
    "visualize_steering_results",
    "generate_steering_examples",
    "gradient_based_vector_search",
    "optimize_bias_direction",
    "visualize_gradient_search",
    # Projection-based steering
    "orthogonal_projection",
    "projection_steering_hook",
    "evaluate_projection_steering",
    "batch_projection_steering_experiment",
    "visualize_projection_steering_results",
    "generate_projection_steering_examples",
    # SAE-based steering
    "SparseAutoencoder",
    "train_sae_on_activations",
    "extract_bias_direction_from_sae",
    "extract_sae_bias_vector",
    "batch_sae_steering_experiment",
]

