import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


DEFAULT_DATASETS = ("stereoset_race", "stereoset_gender", "winogender")
PLOT_DPI = 300
FIGSIZE_SINGLE = (8, 4)
FIGSIZE_COMBINED = (12, 9)


def _ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _load_json(path: Path) -> Optional[Dict[str, float]]:
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)


def _extract_dataset_name(file_path: Path, prefix: str) -> str:
    """Infer dataset name by stripping a known prefix and the `.json` suffix."""
    stem = file_path.stem
    if not stem.startswith(prefix):
        raise ValueError(f"File {file_path} does not start with prefix {prefix}")
    # Remove leading underscore after the prefix, e.g. `linear_probing_`
    dataset = stem[len(prefix) :]
    if dataset.startswith("_"):
        dataset = dataset[1:]
    return dataset


def _list_available_datasets(model_dir: Path) -> List[str]:
    dataset_names = set()
    for prefix in (
        "attribution_patching",
        "head_ablations",
        "mlp_ablations",
        "linear_probing",
    ):
        for file_path in model_dir.glob(f"{prefix}_*.json"):
            try:
                dataset = _extract_dataset_name(file_path, prefix)
                dataset_names.add(dataset)
            except ValueError:
                continue
    return sorted(dataset_names)


def _build_attribution_matrix(
    attribution_scores: Dict[str, float]
) -> Tuple[np.ndarray, List[str], List[str]]:
    """Convert attribution scores into a layer x component matrix."""
    layer_indices: Dict[int, int] = {}
    components: Dict[str, int] = {}

    pattern = re.compile(r"blocks\.(\d+)\.(.+)")
    for hook_name in attribution_scores:
        match = pattern.match(hook_name)
        if not match:
            continue  # skip non-block hooks
        layer = int(match.group(1))
        component = match.group(2)
        layer_indices.setdefault(layer, len(layer_indices))
        components.setdefault(component, len(components))

    if not layer_indices or not components:
        return np.empty((0, 0)), [], []

    matrix = np.full((len(layer_indices), len(components)), np.nan)
    for hook_name, score in attribution_scores.items():
        match = pattern.match(hook_name)
        if not match:
            continue
        layer = int(match.group(1))
        component = match.group(2)
        matrix[layer_indices[layer], components[component]] = score

    layer_labels = [str(idx) for idx, _ in sorted((layer, idx) for layer, idx in layer_indices.items())]
    component_labels = [component for component, _ in sorted(components.items(), key=lambda kv: kv[1])]
    return matrix, layer_labels, component_labels


def _build_head_matrix(head_scores: Dict[str, float]) -> Tuple[np.ndarray, List[str], List[str]]:
    """Convert head ablation scores into a layer x head matrix."""
    if not head_scores:
        return np.empty((0, 0)), [], []

    max_layer = 0
    max_head = 0
    parsed: Dict[Tuple[int, int], float] = {}
    for key, value in head_scores.items():
        if isinstance(key, str) and "_" in key:
            layer_str, head_str = key.split("_", 1)
            try:
                layer = int(layer_str)
                head = int(head_str)
            except ValueError:
                continue
        elif isinstance(key, (tuple, list)) and len(key) == 2:
            layer, head = int(key[0]), int(key[1])
        else:
            continue
        parsed[(layer, head)] = value
        max_layer = max(max_layer, layer)
        max_head = max(max_head, head)

    matrix = np.full((max_layer + 1, max_head + 1), np.nan)
    for (layer, head), score in parsed.items():
        matrix[layer, head] = score

    layer_labels = [str(i) for i in range(max_layer + 1)]
    head_labels = [str(i) for i in range(max_head + 1)]
    return matrix, layer_labels, head_labels


def _build_probe_matrix(
    probe_scores: Dict[str, float], num_layers: Optional[int] = None
) -> Tuple[np.ndarray, List[str], List[str]]:
    if not probe_scores:
        return np.empty((0, 0)), [], []
    entries = []
    for layer_str, score in probe_scores.items():
        try:
            layer = int(layer_str)
        except ValueError:
            continue
        entries.append((layer, score))
    if not entries:
        return np.empty((0, 0)), [], []

    entries.sort(key=lambda x: x[0])
    max_layer = entries[-1][0] if num_layers is None else num_layers
    matrix = np.full((1, max_layer + 1), np.nan)
    for layer, score in entries:
        if layer <= max_layer:
            matrix[0, layer] = score

    layer_labels = [str(i) for i in range(max_layer + 1)]
    return matrix, ["score"], layer_labels


def _setup_heatmap(vmax: Optional[float] = None, diverging: bool = True) -> Dict[str, float]:
    if diverging:
        cmap = "coolwarm"
    else:
        cmap = "viridis"
    return {"cmap": cmap, "vmin": -vmax if diverging and vmax is not None else None, "vmax": vmax}


def _plot_heatmap(
    matrix: np.ndarray,
    row_labels: Sequence[str],
    col_labels: Sequence[str],
    title: str,
    colorbar_label: str,
    output_path: Path,
    diverging: bool,
    annotate: bool = False,
) -> None:
    if matrix.size == 0 or not row_labels or not col_labels:
        print(f"[WARN] Skipping {output_path.name}: no data available.")
        return

    plt.style.use("seaborn-v0_8-darkgrid")
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)

    vmax = np.nanmax(np.abs(matrix)) if diverging else np.nanmax(matrix)
    heatmap_kwargs = _setup_heatmap(vmax, diverging=diverging)
    sns.heatmap(matrix, ax=ax, mask=np.isnan(matrix), xticklabels=col_labels, yticklabels=row_labels, cbar_kws={"label": colorbar_label}, **heatmap_kwargs)

    if annotate:
        for (i, j), val in np.ndenumerate(matrix):
            if np.isnan(val):
                continue
            ax.text(
                j + 0.5,
                i + 0.5,
                f"{val:.2f}",
                ha="center",
                va="center",
                color="white" if abs(val) > (vmax or 0) * 0.5 else "black",
                fontsize=7,
            )

    ax.set_title(title)
    ax.set_xlabel("Component / Head / Layer")
    ax.set_ylabel("Layer index" if len(row_labels) > 1 else "")

    fig.tight_layout()
    fig.savefig(output_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved {output_path}")


def _plot_combined_grid(
    attribution_data: Dict[str, Tuple[np.ndarray, List[str], List[str]]],
    ablation_data: Dict[str, Tuple[np.ndarray, List[str], List[str]]],
    probe_data: Dict[str, Tuple[np.ndarray, List[str], List[str]]],
    datasets: Sequence[str],
    model_name: str,
    output_path: Path,
) -> None:
    plt.style.use("seaborn-v0_8-darkgrid")
    fig, axes = plt.subplots(
        len(datasets),
        3,
        figsize=FIGSIZE_COMBINED,
        squeeze=False,
        constrained_layout=True,
    )

    def _draw(ax, matrix, row_labels, col_labels, title, diverging):
        if matrix.size == 0:
            ax.axis("off")
            ax.set_title(f"{title}\n(no data)")
            return
        vmax = np.nanmax(np.abs(matrix)) if diverging else np.nanmax(matrix)
        heatmap_kwargs = _setup_heatmap(vmax, diverging=diverging)
        sns.heatmap(matrix, ax=ax, mask=np.isnan(matrix), xticklabels=col_labels, yticklabels=row_labels, cbar=False, **heatmap_kwargs)
        ax.set_title(title, fontsize=10)

    for row_idx, dataset in enumerate(datasets):
        _draw(
            axes[row_idx, 0],
            *attribution_data.get(dataset, (np.empty((0, 0)), [], [])),
            title=f"{dataset} – attribution",
            diverging=True,
        )
        _draw(
            axes[row_idx, 1],
            *ablation_data.get(dataset, (np.empty((0, 0)), [], [])),
            title=f"{dataset} – ablation",
            diverging=True,
        )
        _draw(
            axes[row_idx, 2],
            *probe_data.get(dataset, (np.empty((0, 0)), [], [])),
            title=f"{dataset} – head-wise",
            diverging=False,
        )

    fig.suptitle(f"{model_name}: bias analysis overview", fontsize=14)
    fig.savefig(output_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved combined grid to {output_path}")


@dataclass
class ModelDatasetResults:
    model_name: str
    dataset_name: str
    attribution: Optional[Dict[str, float]] = None
    head_ablations: Optional[Dict[str, float]] = None
    probe_scores: Optional[Dict[str, float]] = None


def _load_model_dataset(
    model_dir: Path, dataset: str
) -> ModelDatasetResults:
    attribution = _load_json(model_dir / f"attribution_patching_{dataset}.json")
    head_ablations = _load_json(model_dir / f"head_ablations_{dataset}.json")
    probe = _load_json(model_dir / f"linear_probing_{dataset}.json")
    if probe:
        probe_scores = probe.get("layer_accuracies") or probe.get("layer_aucs")
    else:
        probe_scores = None

    return ModelDatasetResults(
        model_name=model_dir.name,
        dataset_name=dataset,
        attribution=attribution,
        head_ablations=head_ablations,
        probe_scores=probe_scores,
    )


def generate_plots_for_model(
    model_dir: Path,
    datasets: Sequence[str],
) -> None:
    output_dir = model_dir / "figures"
    _ensure_output_dir(output_dir)

    available_datasets = _list_available_datasets(model_dir)
    if not available_datasets:
        print(f"[WARN] No result files found in {model_dir}")
        return

    selected_datasets = [
        ds for ds in datasets if ds in available_datasets
    ] or available_datasets

    attribution_mats = {}
    ablation_mats = {}
    probe_mats = {}

    for dataset in selected_datasets:
        result = _load_model_dataset(model_dir, dataset)

        if result.attribution:
            matrix, row_labels, col_labels = _build_attribution_matrix(result.attribution)
            attribution_mats[dataset] = (matrix, row_labels, col_labels)
            _plot_heatmap(
                matrix=matrix,
                row_labels=row_labels,
                col_labels=col_labels,
                title=f"{result.model_name} – {dataset} attribution patching",
                colorbar_label="Attribution score",
                output_path=output_dir / f"{dataset}_attribution_heatmap.png",
                diverging=True,
            )
        else:
            attribution_mats[dataset] = (np.empty((0, 0)), [], [])

        if result.head_ablations:
            matrix, row_labels, col_labels = _build_head_matrix(result.head_ablations)
            ablation_mats[dataset] = (matrix, row_labels, col_labels)
            _plot_heatmap(
                matrix=matrix,
                row_labels=row_labels,
                col_labels=col_labels,
                title=f"{result.model_name} – {dataset} head ablations",
                colorbar_label="Impact on bias metric",
                output_path=output_dir / f"{dataset}_ablation_heatmap.png",
                diverging=True,
            )
        else:
            ablation_mats[dataset] = (np.empty((0, 0)), [], [])

        if result.probe_scores:
            matrix, row_labels, col_labels = _build_probe_matrix(result.probe_scores)
            probe_mats[dataset] = (matrix, row_labels, col_labels)
            _plot_heatmap(
                matrix=matrix,
                row_labels=row_labels,
                col_labels=col_labels,
                title=f"{result.model_name} – {dataset} head-wise scores",
                colorbar_label="Probe accuracy",
                output_path=output_dir / f"{dataset}_head_scores_heatmap.png",
                diverging=False,
            )
        else:
            probe_mats[dataset] = (np.empty((0, 0)), [], [])

    _plot_combined_grid(
        attribution_data=attribution_mats,
        ablation_data=ablation_mats,
        probe_data=probe_mats,
        datasets=selected_datasets,
        model_name=model_dir.name,
        output_path=output_dir / f"{model_dir.name}_combined_heatmaps.png",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate heatmaps for bias analyses.")
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results"),
        help="Root directory that contains per-model result folders.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=[],
        help="List of model subdirectories to process (defaults to auto-detect).",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=list(DEFAULT_DATASETS),
        help="Datasets to plot (falls back to detected datasets if missing).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_root: Path = args.results_root
    if not results_root.exists():
        raise FileNotFoundError(f"Results root {results_root} does not exist.")

    model_dirs = (
        [results_root / model for model in args.models]
        if args.models
        else [p for p in results_root.iterdir() if p.is_dir()]
    )

    if not model_dirs:
        raise RuntimeError(f"No model directories found under {results_root}")

    for model_dir in model_dirs:
        if not model_dir.exists():
            print(f"[WARN] Skipping missing model directory {model_dir}")
            continue
        print(f"[INFO] Processing model results in {model_dir}")
        generate_plots_for_model(model_dir, datasets=args.datasets)


if __name__ == "__main__":
    main()


