from typing import List, Dict, Any

import numpy as np
import torch
import torch.nn.functional as F
from transformer_lens import HookedTransformer

from .dataset import compute_pronoun_probs
from .suppression import apply_suppression


def compute_gld_metric(model: HookedTransformer, tokenizer, prompts: List[str]) -> float:
    """Compute dataset-level GLD metric."""
    glds = []
    for prompt in prompts:
        male, female = compute_pronoun_probs(model, tokenizer, prompt)
        denom = male + female + 1e-8
        glds.append(abs(male - female) / denom)
    return float(np.mean(glds)) if glds else 0.0


def evaluate_bias_reduction(model: HookedTransformer, tokenizer, prompts: List[str], sae, suppressed_latents: List[int], layer: int, scale: float = 0.0) -> Dict[str, float]:
    """Compute baseline and suppressed GLD metrics."""
    print("Computing baseline GLD...")
    baseline = compute_gld_metric(model, tokenizer, prompts)

    print("Computing GLD after suppression...")
    glds = []
    device = model.cfg.device
    for prompt in prompts:
        tokens = tokenizer.encode(prompt, return_tensors="pt").to(device)
        logits = apply_suppression(
            model=model,
            tokens=tokens,
            sae=sae,
            suppressed_latents=suppressed_latents,
            layer=layer,
            scale=scale,
        )
        next_logits = logits[0, -1, :]
        log_probs = torch.log_softmax(next_logits, dim=-1)
        male_ids = tokenizer.encode(" he", add_special_tokens=False)
        female_ids = tokenizer.encode(" she", add_special_tokens=False)
        male = torch.logsumexp(log_probs[male_ids], dim=0).exp().item()
        female = torch.logsumexp(log_probs[female_ids], dim=0).exp().item()
        denom = male + female + 1e-8
        glds.append(abs(male - female) / denom)
    steered = float(np.mean(glds)) if glds else baseline
    reduction = baseline - steered
    reduction_pct = (reduction / (abs(baseline) + 1e-8)) * 100.0
    return {
        "gld_baseline": baseline,
        "gld_suppressed": steered,
        "gld_reduction": reduction,
        "gld_reduction_percent": reduction_pct,
    }


def evaluate_perplexity(model: HookedTransformer, tokenizer, text_corpus: List[str], sae, suppressed_latents: List[int], layer: int, scale: float = 0.0, max_length: int = 256) -> Dict[str, float]:
    """Compute baseline vs suppressed perplexity on text corpus."""
    device = model.cfg.device
    baseline_losses = []
    steered_losses = []

    for text in text_corpus:
        tokens = tokenizer.encode(text, return_tensors="pt", max_length=max_length, truncation=True)
        if tokens.numel() < 2:
            continue
        tokens = tokens.to(device)

        with torch.no_grad():
            logits = model(tokens)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = tokens[..., 1:].contiguous()
        baseline_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else -100,
        )
        baseline_losses.append(baseline_loss.item())

        logits_suppressed = apply_suppression(model=model, tokens=tokens, sae=sae, suppressed_latents=suppressed_latents, layer=layer, scale=scale)
        shift_logits = logits_suppressed[..., :-1, :].contiguous()
        steered_loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else -100)
        steered_losses.append(steered_loss.item())

    ppl_base = float(np.exp(np.mean(baseline_losses))) if baseline_losses else float("inf")
    ppl_steered = float(np.exp(np.mean(steered_losses))) if steered_losses else float("inf")
    increase = ppl_steered - ppl_base
    increase_pct = (increase / (ppl_base + 1e-8)) * 100.0 if ppl_base != float("inf") else float("inf")
    return {
        "perplexity_baseline": ppl_base,
        "perplexity_suppressed": ppl_steered,
        "perplexity_increase": increase,
        "perplexity_increase_percent": increase_pct,
    }

