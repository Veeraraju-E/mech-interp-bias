"""Dataset loading and preprocessing for StereoSet and WinoGender.

Loads datasets from local JSON files (downloaded via download_datasets.py).
"""

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Any, Tuple
import torch


_STEREO_LABEL_MAP = {0: "unrelated", 1: "antistereotype", 2: "stereotype"}


def _normalize_stereoset_example(raw_item: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize StereoSet entry so downstream code sees consistent keys."""
    context = raw_item.get("context", "")
    bias_type = raw_item.get("bias_type", "unknown")
    target = raw_item.get("target", "")
    item_id = raw_item.get("id", "")
    sentences = raw_item.get("sentences", [])
    
    normalized_sentences = []
    
    if isinstance(sentences, list):
        for sent in sentences:
            if not isinstance(sent, dict):
                continue
            label = sent.get("gold_label", sent.get("label", "unrelated"))
            if isinstance(label, int):
                label = _STEREO_LABEL_MAP.get(label, "unrelated")
            elif isinstance(label, str):
                label = label.lower()
            else:
                label = "unrelated"
            
            normalized_sentences.append({
                "sentence": sent.get("sentence", ""),
                "gold_label": label,
                "id": sent.get("id", "")
            })
    elif isinstance(sentences, dict):
        sentence_list = sentences.get("sentence", [])
        id_list = sentences.get("id", [""] * len(sentence_list))
        label_list = sentences.get("gold_label", sentences.get("labels", []))
        
        for sent, sent_id, label in zip(sentence_list, id_list, label_list):
            if isinstance(label, dict) and "label" in label:
                label = label["label"]
            if isinstance(label, list) and label:
                label = label[0]
            if isinstance(label, int):
                label = _STEREO_LABEL_MAP.get(label, "unrelated")
            elif isinstance(label, str):
                label = label.lower()
            else:
                label = "unrelated"
            
            normalized_sentences.append({
                "sentence": sent,
                "gold_label": label,
                "id": sent_id
            })
    
    return {
        "context": context,
        "bias_type": bias_type,
        "target": target,
        "id": item_id,
        "sentences": normalized_sentences
    }


def load_stereoset(data_dir: Path = None) -> List[Dict[str, Any]]:
    """
    Load StereoSet dataset from local JSON file (Nadeem et al., 2021).
    
    Returns normalized entries that retain bias type metadata so we can
    analyze gender, profession, religion, and nationality biases separately.
    """
    if data_dir is None:
        data_dir = Path("data")
    else:
        data_dir = Path(data_dir)
    
    stereoset_file = data_dir / "stereoset_test.json"
    
    if not stereoset_file.exists():
        raise FileNotFoundError(
            f"StereoSet dataset not found at {stereoset_file}. "
            "Please run download_datasets.py first."
        )
    
    with open(stereoset_file, "r") as f:
        data = json.load(f)
    
    return [_normalize_stereoset_example(item) for item in data]


def load_winogender(data_dir: Path = None) -> List[Dict[str, Any]]:
    """
    Load WinoGender dataset from local JSON file.
    
    Args:
        data_dir: Directory containing the dataset file (default: ./data)
    
    Returns:
        List of examples with all fields preserved from JSON:
        {sentence, profession, pronoun, answer, word, template, example_id,...}
    """
    if data_dir is None:
        data_dir = Path("data")
    else:
        data_dir = Path(data_dir)
    
    winogender_file = data_dir / "winogender_test.json"
    
    if not winogender_file.exists():
        raise FileNotFoundError(
            f"WinoGender dataset not found at {winogender_file}. "
            "Please run download_datasets.py first."
        )
    
    with open(winogender_file, "r") as f:
        examples = json.load(f)
    
    for example in examples:
        if "sentence" not in example:
            example["sentence"] = example.get("text", "")
        if "profession" not in example:
            example["profession"] = example.get("occupation", "")
        if "pronoun" not in example:
            example["pronoun"] = ""
        if "answer" not in example:
            example["answer"] = example.get("correct_answer", example.get("gender", ""))
    
    return examples


def _strip_punctuation(token: str) -> str:
    return token.strip(".,!?;:\"'()[]{}")


def _extract_context_segments(context: str) -> Tuple[str, str]:
    """Return prefix and suffix around BLANK marker (if present)."""
    if "BLANK" not in context:
        return context.strip(), ""
    prefix, *rest = context.split("BLANK", 1)
    suffix = rest[0] if rest else ""
    return prefix.rstrip(), suffix.lstrip()


def build_stereoset_triplets(examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Group normalized StereoSet entries into stereotype/antistereotype triplets."""
    triplets = []
    for item in examples:
        label_map = {}
        for sent in item.get("sentences", []):
            label = sent.get("gold_label", "unrelated")
            label_map[label] = sent.get("sentence", "")
        
        if "stereotype" not in label_map or "antistereotype" not in label_map:
            continue
        
        prefix, suffix = _extract_context_segments(item.get("context", ""))
        
        triplet = {
            "context": item.get("context", ""),
            "context_prefix": prefix,
            "context_suffix": suffix,
            "bias_type": item.get("bias_type", "unknown"),
            "target": item.get("target", ""),
            "stereotype_sentence": label_map["stereotype"],
            "antistereotype_sentence": label_map["antistereotype"]
        }
        if "unrelated" in label_map:
            triplet["unrelated_sentence"] = label_map["unrelated"]
        triplets.append(triplet)
    return triplets


def find_first_difference_tokens(sentence_a: str, sentence_b: str) -> Tuple[str, str]:
    """Return the first pair of differing word tokens across two sentences."""
    a_tokens = sentence_a.split()
    b_tokens = sentence_b.split()
    
    for a_tok, b_tok in zip(a_tokens, b_tokens):
        if a_tok != b_tok:
            return _strip_punctuation(a_tok), _strip_punctuation(b_tok)
    
    # Fallback to last token if sentences differ only by length
    if a_tokens and b_tokens:
        return _strip_punctuation(a_tokens[-1]), _strip_punctuation(b_tokens[-1])
    return "", ""


def build_winogender_pairs(examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Group WinoGender samples into male/female pairs sharing the same template."""
    groups = defaultdict(lambda: {"male": [], "female": []})
    for ex in examples:
        answer = ex.get("answer", "").lower()
        if answer not in {"male", "female"}:
            continue
        groups[ex.get("example_id", "")][answer].append(ex)
    
    pairs = []
    for example_id, data in groups.items():
        if not data["male"] or not data["female"]:
            continue
        for male_ex, female_ex in zip(data["male"], data["female"]):
            template = male_ex.get("template") or female_ex.get("template") or ""
            prompt = template
            match = re.split(r"\{pronoun[0-9]*\}", template)
            if match:
                prompt = match[0].strip()
            else:
                sentence = male_ex.get("sentence", "")
                pronoun = male_ex.get("pronoun", "")
                idx = sentence.lower().find(pronoun.lower())
                prompt = sentence[:idx].strip() if idx >= 0 else sentence
            
            pairs.append({
                "example_id": example_id,
                "prompt": prompt,
                "profession": male_ex.get("profession", ""),
                "word": male_ex.get("word", ""),
                "male_sentence": male_ex.get("sentence", ""),
                "female_sentence": female_ex.get("sentence", ""),
                "male_pronoun": male_ex.get("pronoun", "he"),
                "female_pronoun": female_ex.get("pronoun", "she")
            })
    return pairs


def prepare_prompts(examples: List[Dict[str, Any]], tokenizer, max_length: int = 128) -> List[torch.Tensor]:
    """
    Tokenize prompts for GPT-2.
    
    Args:
        examples: List of example dictionaries
        tokenizer: GPT-2 tokenizer
        max_length: Maximum sequence length
    
    Returns:
        List of tokenized input tensors
    """
    tokenized = []
    for example in examples:
        # For StereoSet, use context + sentence
        if "context" in example:
            text = f"{example['context']} {example['sentence']}"
        else:
            # For WinoGender, use sentence
            text = example["sentence"]
        
        encoded = tokenizer(
            text,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding="max_length"
        )
        tokenized.append(encoded["input_ids"].squeeze(0))
    
    return tokenized


def get_stereoset_pairs(examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Group StereoSet examples into stereotype/antistereotype pairs.
    
    Args:
        examples: List of StereoSet examples
    
    Returns:
        List of pairs enriched with bias metadata.
    """
    pairs = []
    for triplet in build_stereoset_triplets(examples):
        pair = {
            "context": triplet["context"],
            "context_prefix": triplet["context_prefix"],
            "context_suffix": triplet["context_suffix"],
            "bias_type": triplet["bias_type"],
            "target": triplet["target"],
            "stereotype_sentence": triplet["stereotype_sentence"],
            "antistereotype_sentence": triplet["antistereotype_sentence"]
        }
        if "unrelated_sentence" in triplet:
            pair["unrelated_sentence"] = triplet["unrelated_sentence"]
        pairs.append(pair)
    return pairs


def load_acdc_pairs(dataset_type: str, data_dir: Path = None) -> List[Dict[str, Any]]:
    """
    Load ACDC-formatted pairs for activation patching experiments.
    
    Args:
        dataset_type: One of "stereoset_gender", "stereoset_race", or "winogender"
        data_dir: Directory containing the ACDC pair files (default: ./data)
    
    Returns:
        List of ACDC pairs with clean/corrupted sentence pairs and metadata
    """
    if data_dir is None:
        data_dir = Path("data")
    else:
        data_dir = Path(data_dir)
    
    file_map = {
        "stereoset_gender": "stereoset_gender_acdc_pairs.json",
        "stereoset_race": "stereoset_race_acdc_pairs.json",
        "winogender": "winogender_acdc_pairs.json"
    }
    
    if dataset_type not in file_map:
        raise ValueError(
            f"Unknown dataset_type: {dataset_type}. "
            f"Must be one of {list(file_map.keys())}"
        )
    
    file_path = data_dir / file_map[dataset_type]
    
    if not file_path.exists():
        raise FileNotFoundError(
            f"ACDC pairs file not found at {file_path}. "
            f"Please ensure the file exists in the data directory."
        )
    
    with open(file_path, "r") as f:
        pairs = json.load(f)
    
    return pairs


def get_acdc_stereoset_pairs(bias_type: str = "gender", data_dir: Path = None) -> List[Dict[str, Any]]:
    """
    Get ACDC-formatted StereoSet pairs for a specific bias type.
    
    Args:
        bias_type: Either "gender" or "race"
        data_dir: Directory containing the ACDC pair files
    
    Returns:
        List of clean/corrupted pairs for probing and interventions
    """
    dataset_type = f"stereoset_{bias_type}"
    return load_acdc_pairs(dataset_type, data_dir)


def get_acdc_winogender_pairs(data_dir: Path = None) -> List[Dict[str, Any]]:
    """
    Get ACDC-formatted WinoGender pairs.
    
    Args:
        data_dir: Directory containing the ACDC pair files
    
    Returns:
        List of clean/corrupted pairs with male/female pronoun swaps
    """
    return load_acdc_pairs("winogender", data_dir)
