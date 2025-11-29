import torch
import torch.nn.functional as F
from typing import List, Dict, Any, Sequence
from transformer_lens import HookedTransformer
from transformers import GPT2Tokenizer

from .data_loader import *


def compute_stereoset_score(model: HookedTransformer, examples: List[Dict[str, Any]], tokenizer: GPT2Tokenizer) -> float:
    triplets = build_stereoset_triplets(examples)
    if not triplets:
        return 0.0
    
    scores = []
    device = model.cfg.device
    for triplet in triplets:
        prompt = triplet["context_prefix"] or triplet["context"]
        if not prompt:
            continue
        tokens = tokenizer.encode(prompt, return_tensors="pt").to(device)
        if tokens.numel() == 0:
            continue
        
        stereo_word, anti_word = find_first_difference_tokens(
            triplet["stereotype_sentence"],
            triplet["antistereotype_sentence"]
        )
        if not stereo_word or not anti_word:
            continue
        
        stereo_token_ids = tokenizer.encode(stereo_word, add_special_tokens=False)
        antistereo_token_ids = tokenizer.encode(anti_word, add_special_tokens=False)
        if not stereo_token_ids or not antistereo_token_ids:
            continue
        
        with torch.no_grad():
            logits = model(tokens)
            next_token_logits = logits[0, -1, :] if logits.dim() == 3 else logits[-1, :]
            log_probs = F.log_softmax(next_token_logits, dim=-1)
            stereo_lp = torch.logsumexp(log_probs[stereo_token_ids], dim=0)
            anti_lp = torch.logsumexp(log_probs[antistereo_token_ids], dim=0)
            scores.append((stereo_lp - anti_lp).item())
    
    return sum(scores) / len(scores) if scores else 0.0


def compute_winogender_score(model: HookedTransformer, examples: List[Dict[str, Any]], tokenizer: GPT2Tokenizer) -> float:
    pairs = build_winogender_pairs(examples)
    if not pairs:
        return 0.0
    
    device = model.cfg.device
    scores = []
    for pair in pairs:
        prompt = pair["prompt"]
        if not prompt:
            continue
        tokens = tokenizer.encode(prompt, return_tensors="pt").to(device)
        if tokens.numel() == 0:
            continue
        
        male_tokens = tokenizer.encode(pair["male_pronoun"], add_special_tokens=False)
        female_tokens = tokenizer.encode(pair["female_pronoun"], add_special_tokens=False)
        if not male_tokens or not female_tokens:
            continue
        
        with torch.no_grad():
            logits = model(tokens)
            next_token_logits = logits[0, -1, :] if logits.dim() == 3 else logits[-1, :]
            log_probs = F.log_softmax(next_token_logits, dim=-1)
            male_lp = torch.logsumexp(log_probs[male_tokens], dim=0)
            female_lp = torch.logsumexp(log_probs[female_tokens], dim=0)
            scores.append((male_lp - female_lp).item())
    
    return sum(scores) / len(scores) if scores else 0.0


def _logprob_difference(log_probs: torch.Tensor, positive_ids: Sequence[int], negative_ids: Sequence[int]) -> torch.Tensor:
    if not positive_ids or not negative_ids:
        return torch.tensor(0.0, device=log_probs.device, requires_grad=True)
    
    pos = torch.tensor(positive_ids, device=log_probs.device, dtype=torch.long)
    neg = torch.tensor(negative_ids, device=log_probs.device, dtype=torch.long)
    pos_lp = torch.logsumexp(log_probs.index_select(0, pos), dim=0)
    neg_lp = torch.logsumexp(log_probs.index_select(0, neg), dim=0)
    return pos_lp - neg_lp


def build_bias_metric_fn(dataset_name: str):
    def metric_fn(logits: torch.Tensor, metadata: Dict[str, Any]) -> torch.Tensor:
        if logits.dim() == 3:
            next_token_logits = logits[0, -1, :]
        else:
            next_token_logits = logits[-1, :]
        log_probs = F.log_softmax(next_token_logits, dim=-1)
        
        if dataset_name == "stereoset":
            return _logprob_difference(
                log_probs,
                metadata.get("stereo_token_ids", []),
                metadata.get("antistereo_token_ids", [])
            )
        elif dataset_name == "winogender":
            return _logprob_difference(
                log_probs,
                metadata.get("male_token_ids", []),
                metadata.get("female_token_ids", [])
            )
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return metric_fn


def compute_bias_metric(model: HookedTransformer, examples: List[Dict[str, Any]], dataset_name: str, tokenizer: GPT2Tokenizer) -> float:
    if dataset_name.lower() == "stereoset":
        return compute_stereoset_score(model, examples, tokenizer)
    elif dataset_name.lower() == "winogender":
        return compute_winogender_score(model, examples, tokenizer)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
