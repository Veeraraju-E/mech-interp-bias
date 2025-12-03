import torch
import torch.nn.functional as F
from typing import List, Dict, Any, Sequence, Union
from transformer_lens import HookedTransformer
from transformers import GPT2Tokenizer, PreTrainedTokenizerBase

from .data_loader import *


def compute_stereoset_score(model: HookedTransformer, examples: List[Dict[str, Any]], tokenizer: Union[GPT2Tokenizer, PreTrainedTokenizerBase]) -> float:
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


def compute_winogender_score(model: HookedTransformer, examples: List[Dict[str, Any]], tokenizer: Union[GPT2Tokenizer, PreTrainedTokenizerBase]) -> float:
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


def _normalize_token_ids(token_ids: Any) -> List[int]:
    """Convert token_ids to a list, handling both lists and tensors."""
    if token_ids is None:
        return []
    if isinstance(token_ids, torch.Tensor):
        if token_ids.numel() == 0:
            return []
        # Detach and move to CPU to avoid any gradient/device issues
        return token_ids.detach().cpu().tolist()
    if isinstance(token_ids, (list, tuple)):
        # Ensure all elements are integers
        try:
            return [int(x) for x in token_ids]
        except (ValueError, TypeError):
            return []
    # Try to convert other types (e.g., numpy arrays)
    try:
        return [int(x) for x in token_ids]
    except (ValueError, TypeError):
        return []


def _logprob_difference(log_probs: torch.Tensor, positive_ids: Sequence[int], negative_ids: Sequence[int]) -> torch.Tensor:
    # Normalize to lists first to avoid any tensor boolean issues
    pos_list = _normalize_token_ids(positive_ids)
    neg_list = _normalize_token_ids(negative_ids)
    
    if not isinstance(pos_list, list):
        pos_list = list(pos_list)
    if not isinstance(neg_list, list):
        neg_list = list(neg_list)
    
    if len(pos_list) == 0 or len(neg_list) == 0:
        raise ValueError("Positive or negative IDs are empty")
    
    pos = torch.tensor(pos_list, device=log_probs.device, dtype=torch.long)
    neg = torch.tensor(neg_list, device=log_probs.device, dtype=torch.long)
    
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


def compute_bias_metric(model: HookedTransformer, examples: List[Dict[str, Any]], dataset_name: str, tokenizer: Union[GPT2Tokenizer, PreTrainedTokenizerBase]) -> float:
    if dataset_name.lower() == "stereoset":
        return compute_stereoset_score(model, examples, tokenizer)
    elif dataset_name.lower() == "winogender":
        return compute_winogender_score(model, examples, tokenizer)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")