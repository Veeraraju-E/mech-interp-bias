"""Probing and layer analysis methods for mechanistic interpretability."""

from .linear_probes import (
    LinearProbe,
    train_layerwise_probes,
    evaluate_probe,
    plot_probe_results
)

from .logit_lens import (
    apply_logit_lens,
    compute_layerwise_bias,
    analyze_bias_emergence,
    visualize_logit_lens
)

from .interventions import (
    ablate_layers,
    ablate_heads,
    evaluate_layer_ablation,
    find_biased_heads,
    compute_perplexity,
    visualize_intervention_results,
    load_attribution_results,
    evaluate_attribution_guided_ablation
)

from .enhanced_viz import (
    plot_probe_roc_curves,
    plot_confusion_matrix,
    plot_head_layer_heatmap,
    plot_comparison_dashboard,
    plot_metric_heatmap,
    plot_pareto_frontier,
    create_statistical_summary,
    plot_activation_clustering,
    plot_token_probability_heatmap,
    plot_attention_flow
)

__all__ = [
    "LinearProbe",
    "train_layerwise_probes",
    "evaluate_probe",
    "plot_probe_results",
    "apply_logit_lens",
    "compute_layerwise_bias",
    "analyze_bias_emergence",
    "visualize_logit_lens",
    "ablate_layers",
    "ablate_heads",
    "evaluate_layer_ablation",
    "find_biased_heads",
    "compute_perplexity",
    "visualize_intervention_results",
    "load_attribution_results",
    "evaluate_attribution_guided_ablation",
    "plot_probe_roc_curves",
    "plot_confusion_matrix",
    "plot_head_layer_heatmap",
    "plot_comparison_dashboard",
    "plot_metric_heatmap",
    "plot_pareto_frontier",
    "create_statistical_summary",
    "plot_activation_clustering",
    "plot_token_probability_heatmap",
    "plot_attention_flow",
]

