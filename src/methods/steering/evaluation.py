from typing import List, Dict, Any

import numpy as np
import torch
import torch.nn.functional as F
from transformer_lens import HookedTransformer

from .dataset import compute_pronoun_probs
from .suppression import apply_suppression


def compute_gld_metric(model: HookedTransformer, tokenizer, prompts: List[str]) -> float:
    """Compute dataset-level GLD metric."""
    glds = [
        abs(male - female) / (male + female + 1e-8)
        for prompt in prompts
        for male, female in [compute_pronoun_probs(model, tokenizer, prompt)]
    ]
    return float(np.mean(glds)) if glds else 0.0


def evaluate_bias_reduction(model: HookedTransformer, tokenizer, prompts: List[str], sae, suppressed_latents: List[int], layer: int, scale: float = 0.0) -> Dict[str, float]:
    """Compute baseline and suppressed GLD metrics."""
    print("Computing baseline GLD...")
    baseline = compute_gld_metric(model, tokenizer, prompts)

    print("Computing GLD after suppression...")
    device = model.cfg.device
    male_ids = tokenizer.encode(" he", add_special_tokens=False)
    female_ids = tokenizer.encode(" she", add_special_tokens=False)
    glds = []
    for prompt in prompts:
        tokens = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
        logits = apply_suppression(model, tokens, sae, suppressed_latents, layer, scale)
        log_probs = torch.log_softmax(logits[0, -1, :], dim=-1)
        male = torch.logsumexp(log_probs[male_ids], dim=0).exp().item()
        female = torch.logsumexp(log_probs[female_ids], dim=0).exp().item()
        glds.append(abs(male - female) / (male + female + 1e-8))
    
    steered = float(np.mean(glds)) if glds else baseline
    reduction = baseline - steered
    return {
        "gld_baseline": baseline, "gld_suppressed": steered, "gld_reduction": reduction,
        "gld_reduction_percent": (reduction / (abs(baseline) + 1e-8)) * 100.0,
    }


def evaluate_perplexity(model: HookedTransformer, tokenizer, text_corpus: List[str], sae, suppressed_latents: List[int], layer: int, scale: float = 0.0, max_length: int = 256) -> Dict[str, float]:
    """Compute baseline vs suppressed perplexity on text corpus."""
    device = model.cfg.device
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else -100
    baseline_losses, steered_losses = [], []

    for text in text_corpus:
        tokens = tokenizer(text, return_tensors="pt", max_length=max_length, truncation=True)["input_ids"]
        if tokens.numel() < 2:
            continue
        tokens = tokens.to(device)
        shift_labels = tokens[..., 1:].contiguous()

        with torch.no_grad():
            baseline_logits = model(tokens)
            baseline_loss = F.cross_entropy(
                baseline_logits[..., :-1, :].contiguous().view(-1, baseline_logits.size(-1)),
                shift_labels.view(-1), ignore_index=pad_id
            )
            baseline_losses.append(baseline_loss.item())

            steered_logits = apply_suppression(model, tokens, sae, suppressed_latents, layer, scale)
            steered_loss = F.cross_entropy(
                steered_logits[..., :-1, :].contiguous().view(-1, steered_logits.size(-1)),
                shift_labels.view(-1), ignore_index=pad_id
            )
            steered_losses.append(steered_loss.item())

    ppl_base = float(np.exp(np.mean(baseline_losses))) if baseline_losses else float("inf")
    ppl_steered = float(np.exp(np.mean(steered_losses))) if steered_losses else float("inf")
    increase = ppl_steered - ppl_base
    return {
        "perplexity_baseline": ppl_base, "perplexity_suppressed": ppl_steered,
        "perplexity_increase": increase, "perplexity_increase_percent": (increase / (ppl_base + 1e-8)) * 100.0 if ppl_base != float("inf") else float("inf"),
    }

