"""Caching utilities for expensive computations in bias analysis methods."""

import json
import pickle
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional, List
import numpy as np
import torch


def get_model_name(model) -> str:
    """Get model name from HookedTransformer model."""
    if hasattr(model, 'cfg'):
        if hasattr(model.cfg, 'model_name'):
            return model.cfg.model_name
        if hasattr(model.cfg, 'name'):
            return model.cfg.name
    return "gpt2-medium"


def get_cache_key(model_name: str, dataset_name: str, method: str, **kwargs) -> str:
    """Generate a cache key from model, dataset, method, and additional parameters."""
    key_parts = [model_name, dataset_name, method]
    for k, v in sorted(kwargs.items()):
        if isinstance(v, (str, int, float, bool)):
            key_parts.append(f"{k}={v}")
        elif isinstance(v, (list, tuple)):
            key_parts.append(f"{k}={','.join(map(str, v))}")
    key_str = "_".join(key_parts)
    return hashlib.md5(key_str.encode()).hexdigest()


def save_cache(cache_dir: Path, cache_key: str, data: Any, suffix: str = ".pkl"):
    """Save data to cache directory."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{cache_key}{suffix}"
    
    if suffix == ".pkl":
        with open(cache_file, "wb") as f:
            pickle.dump(data, f)
    elif suffix == ".json":
        with open(cache_file, "w") as f:
            json.dump(data, f, indent=2)
    elif suffix == ".npz":
        if isinstance(data, dict):
            np.savez_compressed(cache_file, **data)
        else:
            np.savez_compressed(cache_file, data=data)
    
    return cache_file


def load_cache(cache_dir: Path, cache_key: str, suffix: str = ".pkl") -> Optional[Any]:
    """Load data from cache directory."""
    cache_file = cache_dir / f"{cache_key}{suffix}"
    
    if not cache_file.exists():
        return None
    
    try:
        if suffix == ".pkl":
            with open(cache_file, "rb") as f:
                return pickle.load(f)
        elif suffix == ".json":
            with open(cache_file, "r") as f:
                return json.load(f)
        elif suffix == ".npz":
            return dict(np.load(cache_file, allow_pickle=True))
    except Exception as e:
        print(f"Warning: Failed to load cache {cache_file}: {e}")
        return None


def cache_activations(
    cache_dir: Path,
    model_name: str,
    dataset_name: str,
    activations: Dict[int, np.ndarray],
    labels: np.ndarray,
    position: str = "last",
    use_cache: bool = True
):
    """Cache activations for linear probing."""
    if not use_cache:
        return
    
    cache_key = get_cache_key(model_name, dataset_name, "linear_probing_activations", position=position)
    print(f"Caching activations to {cache_dir}")
    cache_data = {str(k): v for k, v in activations.items()}
    cache_data["labels"] = labels
    save_cache(cache_dir, cache_key, cache_data, suffix=".npz")


def load_cached_activations(
    cache_dir: Path,
    model_name: str,
    dataset_name: str,
    position: str = "last"
) -> Optional[tuple]:
    """Load cached activations and labels."""
    cache_key = get_cache_key(model_name, dataset_name, "linear_probing_activations", position=position)
    cached = load_cache(cache_dir, cache_key, suffix=".npz")
    
    if cached is None:
        return None
    
    activations_dict = {}
    for k, v in cached.items():
        if k != "labels":
            try:
                layer_idx = int(k)
                activations_dict[layer_idx] = v
            except (ValueError, TypeError):
                continue
    
    labels_array = cached.get("labels", None)
    if labels_array is not None:
        labels_array = np.array(labels_array)
    
    return activations_dict, labels_array


def cache_prepared_examples(
    cache_dir: Path,
    model_name: str,
    dataset_name: str,
    prepared_examples: List[Dict[str, Any]],
    use_cache: bool = True
) -> Optional[List[Dict[str, Any]]]:
    """Cache or load prepared examples for activation patching."""
    if not use_cache:
        return None
    
    cache_key = get_cache_key(model_name, dataset_name, "activation_patching_examples")
    cached = load_cache(cache_dir, cache_key, suffix=".pkl")
    
    if cached is not None:
        print(f"Loading cached prepared examples from {cache_dir}")
        return cached
    
    print(f"Caching prepared examples to {cache_dir}")
    save_cache(cache_dir, cache_key, prepared_examples, suffix=".pkl")
    return None


def load_cached_prepared_examples(
    cache_dir: Path,
    model_name: str,
    dataset_name: str
) -> Optional[List[Dict[str, Any]]]:
    """Load cached prepared examples."""
    cache_key = get_cache_key(model_name, dataset_name, "activation_patching_examples")
    return load_cache(cache_dir, cache_key, suffix=".pkl")


def cache_bias_scores(
    cache_dir: Path,
    model_name: str,
    dataset_name: str,
    scores: Dict[str, float],
    use_cache: bool = True
) -> Optional[Dict[str, float]]:
    """Cache or load bias scores."""
    if not use_cache:
        return None
    
    cache_key = get_cache_key(model_name, dataset_name, "bias_scores")
    cached = load_cache(cache_dir, cache_key, suffix=".json")
    
    if cached is not None:
        print(f"Loading cached bias scores from {cache_dir}")
        return cached
    
    print(f"Caching bias scores to {cache_dir}")
    save_cache(cache_dir, cache_key, scores, suffix=".json")
    return None


def load_cached_bias_scores(
    cache_dir: Path,
    model_name: str,
    dataset_name: str
) -> Optional[Dict[str, float]]:
    """Load cached bias scores."""
    cache_key = get_cache_key(model_name, dataset_name, "bias_scores")
    return load_cache(cache_dir, cache_key, suffix=".json")


def cache_attribution_results(
    cache_dir: Path,
    model_name: str,
    dataset_name: str,
    attributions: Dict[str, float],
    use_cache: bool = True
) -> Optional[Dict[str, float]]:
    """Cache or load attribution patching results."""
    if not use_cache:
        return None
    
    cache_key = get_cache_key(model_name, dataset_name, "attribution_patching")
    cached = load_cache(cache_dir, cache_key, suffix=".json")
    
    if cached is not None:
        print(f"Loading cached attribution results from {cache_dir}")
        return cached
    
    print(f"Caching attribution results to {cache_dir}")
    save_cache(cache_dir, cache_key, attributions, suffix=".json")
    return None


def load_cached_attribution_results(
    cache_dir: Path,
    model_name: str,
    dataset_name: str
) -> Optional[Dict[str, float]]:
    """Load cached attribution patching results."""
    cache_key = get_cache_key(model_name, dataset_name, "attribution_patching")
    return load_cache(cache_dir, cache_key, suffix=".json")


def cache_ablation_results(
    cache_dir: Path,
    model_name: str,
    dataset_name: str,
    head_impacts: Dict[Any, float],
    mlp_impacts: Dict[int, float],
    use_cache: bool = True
) -> Optional[tuple]:
    """Cache or load ablation results."""
    if not use_cache:
        return None
    
    head_key = get_cache_key(model_name, dataset_name, "head_ablations")
    mlp_key = get_cache_key(model_name, dataset_name, "mlp_ablations")
    
    cached_heads = load_cache(cache_dir, head_key, suffix=".json")
    cached_mlps = load_cache(cache_dir, mlp_key, suffix=".json")
    
    if cached_heads is not None and cached_mlps is not None:
        print(f"Loading cached ablation results from {cache_dir}")
        return cached_heads, cached_mlps
    
    print(f"Caching ablation results to {cache_dir}")
    save_cache(cache_dir, head_key, head_impacts, suffix=".json")
    save_cache(cache_dir, mlp_key, mlp_impacts, suffix=".json")
    return None


def load_cached_ablation_results(
    cache_dir: Path,
    model_name: str,
    dataset_name: str
) -> Optional[tuple]:
    """Load cached ablation results."""
    head_key = get_cache_key(model_name, dataset_name, "head_ablations")
    mlp_key = get_cache_key(model_name, dataset_name, "mlp_ablations")
    
    cached_heads = load_cache(cache_dir, head_key, suffix=".json")
    cached_mlps = load_cache(cache_dir, mlp_key, suffix=".json")
    
    if cached_heads is not None and cached_mlps is not None:
        return cached_heads, cached_mlps
    
    return None

