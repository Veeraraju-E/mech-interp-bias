from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import torch
from tqdm import tqdm

from src.data_loader import *
from src.methods.linear_probing.probe import collect_activations


@dataclass
class PromptExample:
    """Container for a neutral prompt and metadata."""

    prompt: str
    source: str
    metadata: Dict[str, Any]


def _build_stereoset_prompts(max_examples: Optional[int] = None) -> List[PromptExample]:
    pairs = [p for p in get_stereoset_pairs(load_stereoset()) if p.get("bias_type", "").lower() == "gender"]
    prompts = []
    for pair in pairs[:max_examples] if max_examples else pairs:
        context = (pair.get("context_prefix") or pair.get("context") or "").strip()
        if context:
            prompts.append(PromptExample(
                prompt=context,
                source="stereoset",
                metadata={"target": pair.get("target", ""), "context_suffix": pair.get("context_suffix", "")},
            ))
    return prompts


def _build_winogender_prompts(max_examples: Optional[int] = None) -> List[PromptExample]:
    pairs = build_winogender_pairs(load_winogender())
    prompts = []
    for pair in pairs[:max_examples] if max_examples else pairs:
        prompt = pair.get("prompt", "").strip()
        if prompt:
            prompts.append(PromptExample(
                prompt=prompt,
                source="winogender",
                metadata={"profession": pair.get("profession", ""), "word": pair.get("word", "")},
            ))
    return prompts


def build_gender_prompts(dataset_name: str, max_examples: Optional[int] = None) -> List[PromptExample]:
    """Return neutral prompts for a dataset."""
    dataset_name = dataset_name.lower()
    if dataset_name == "stereoset_gender": return _build_stereoset_prompts(max_examples)
    if dataset_name == "winogender": return _build_winogender_prompts(max_examples)
    raise ValueError(f"Unsupported dataset for GLD prompts: {dataset_name}")

def _token_ids(tokenizer, text: str) -> List[int]:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if not ids:
        raise ValueError(f"Tokenizer returned empty ids for text: {text!r}")
    return ids


def compute_pronoun_probs(model, tokenizer, prompt: str) -> Tuple[float, float]:
    """Return probabilities for 'he' and 'she' conditioned on prompt."""
    device = model.cfg.device
    tokens = tokenizer(prompt, return_tensors="pt")["input_ids"]
    tokens = tokens.to(device)
    with torch.no_grad():
        logits = model(tokens)
    next_logits = logits[0, -1, :]
    log_probs = torch.log_softmax(next_logits, dim=-1)

    male_ids = _token_ids(tokenizer, " he")
    female_ids = _token_ids(tokenizer, " she")
    male_lp = torch.logsumexp(log_probs[male_ids], dim=0).exp().item()
    female_lp = torch.logsumexp(log_probs[female_ids], dim=0).exp().item()
    return male_lp, female_lp


def compute_gld_scores(model, tokenizer, prompts: List[PromptExample]) -> List[Dict[str, Any]]:
    scores = []
    for example in tqdm(prompts, desc="Computing GLD"):
        try:
            male_prob, female_prob = compute_pronoun_probs(model, tokenizer, example.prompt)
            gld = abs(male_prob - female_prob) / (male_prob + female_prob + 1e-8)
            scores.append({
                "prompt": example.prompt, "gld": gld, "male_prob": male_prob, "female_prob": female_prob,
                "source": example.source, "metadata": example.metadata,
            })
        except ValueError:
            continue
    return scores


def collect_activation_dataset(model, tokenizer, dataset_name: str, layer: int, max_examples: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    """
    Collect residual activations and GLD labels for training SAEs.

    Returns:
        activations: [n_samples, hidden_dim]
        gld_scores: [n_samples]
        metadata: list of dict (prompt info + GLD stats)
    """
    prompts = build_gender_prompts(dataset_name, max_examples)
    if not prompts:
        raise ValueError(f"No prompts available for dataset {dataset_name}")

    scored_entries = compute_gld_scores(model, tokenizer, prompts)
    if not scored_entries:
        raise RuntimeError("Failed to compute GLD scores for prompts.")

    tokenized = [
        tokenizer(entry["prompt"], return_tensors="pt", max_length=128, truncation=True)["input_ids"].squeeze(0)
        for entry in scored_entries
    ]

    activations_dict = collect_activations(model, tokenized, layer_indices=[layer], position="last")
    activations = activations_dict[layer]
    gld_scores = np.array([entry["gld"] for entry in scored_entries], dtype=np.float32)
    return activations, gld_scores, scored_entries