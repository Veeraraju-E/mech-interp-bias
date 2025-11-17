"""Dataset loading and preprocessing for StereoSet and WinoGender.

Loads datasets from local JSON files (downloaded via download_datasets.py).
"""

import json
from pathlib import Path
from typing import List, Dict, Any
import torch


def load_stereoset(data_dir: Path = None) -> List[Dict[str, Any]]:
    """
    Load StereoSet dataset from local JSON file.
    
    Args:
        data_dir: Directory containing the dataset file (default: ./data)
    
    Returns:
        List of examples with format: {context, sentence, label}
        where label is one of: 'stereotype', 'antistereotype', 'unrelated'
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
    
    examples = []
    for item in data:
        context = item["context"]
        sentences = item["sentences"]
        
        # Handle different possible structures
        if isinstance(sentences, dict):
            # New structure: sentences is a dict with 'sentence' and 'gold_label' lists
            sentence_list = sentences.get("sentence", [])
            label_list = sentences.get("gold_label", [])
            
            for sent, label in zip(sentence_list, label_list):
                # Convert numeric labels to string labels
                # 0 = unrelated, 1 = antistereotype, 2 = stereotype
                if isinstance(label, int):
                    label_map = {0: "unrelated", 1: "antistereotype", 2: "stereotype"}
                    label = label_map.get(label, "unrelated")
                elif isinstance(label, str):
                    # Already a string label
                    pass
                else:
                    label = "unrelated"
                
                examples.append({
                    "context": context,
                    "sentence": sent,
                    "label": label
                })
        else:
            # Old structure: sentences is a list of dicts
            for sent in sentences:
                if isinstance(sent, dict):
                    sentence_text = sent.get("sentence", "")
                    label = sent.get("gold_label", "unrelated")
                    examples.append({
                        "context": context,
                        "sentence": sentence_text,
                        "label": label
                    })
    
    return examples


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
        List of pairs: {context, stereotype_sentence, antistereotype_sentence}
    """
    # Group by context
    context_groups = {}
    for ex in examples:
        ctx = ex["context"]
        if ctx not in context_groups:
            context_groups[ctx] = {}
        label = ex["label"]
        if label in ["stereotype", "antistereotype"]:
            context_groups[ctx][label] = ex["sentence"]
    
    pairs = []
    for ctx, labels in context_groups.items():
        if "stereotype" in labels and "antistereotype" in labels:
            pairs.append({
                "context": ctx,
                "stereotype_sentence": labels["stereotype"],
                "antistereotype_sentence": labels["antistereotype"]
            })
    
    return pairs
