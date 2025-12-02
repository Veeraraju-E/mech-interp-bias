from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
from transformer_lens import HookedTransformer
from transformers import PreTrainedTokenizerBase

DEFAULT_MMLU_SUBJECTS: Sequence[str] = ("abstract_algebra", "anatomy", "college_biology", "moral_scenarios", "professional_law", "us_foreign_policy")
MMLU_DATASET_IDS: Sequence[str] = ("cais/mmlu", "lukaemon/mmlu")


def _load_subject_dataset(subject: str, split: str):
    last_error: Optional[Exception] = None
    for dataset_id in MMLU_DATASET_IDS:
        try:
            return load_dataset(dataset_id, subject, split=split)
        except Exception as err:
            last_error = err
    raise RuntimeError(f"Unable to load MMLU subject '{subject}' from {MMLU_DATASET_IDS}: {last_error}")


def load_mmlu_samples(subjects: Sequence[str], split: str = "test", max_questions_per_subject: int = 25) -> List[Dict[str, str]]:
    """Load a small slice of MMLU questions for fast experimentation."""
    samples: List[Dict[str, str]] = []
    for subject in subjects:
        dataset = _load_subject_dataset(subject, split)
        limit = min(max_questions_per_subject, len(dataset))
        for example in dataset.select(range(limit)):
            samples.append(
                {
                    "subject": subject,
                    "question": example["question"],
                    "choices": example["choices"],
                    "answer": example["answer"],
                }
            )
    return samples


def _format_prompt(question: str, choices: Sequence[str]) -> str:
    """Format prompt with explicit answer choices."""
    labels = ["A", "B", "C", "D"]
    choice_lines = [f"{label}. {choice}" for label, choice in zip(labels, choices)]
    return f"{question.strip()}\n" + "\n".join(choice_lines) + "\nAnswer:"


def _run_with_hooks(model: HookedTransformer, tokens: torch.Tensor, steering_hooks: Optional[List[tuple]] = None) -> torch.Tensor:
    if steering_hooks:
        return model.run_with_hooks(tokens, fwd_hooks=steering_hooks)
    return model(tokens)


def _choice_logprob(logits: torch.Tensor, tokens: torch.Tensor, prompt_len: int) -> float:
    """Compute log probability of the answer suffix."""
    log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
    return sum(log_probs[0, idx - 1, tokens[0, idx]].item() for idx in range(prompt_len, tokens.shape[1]))


def evaluate_mmlu_accuracy(model: HookedTransformer, tokenizer: PreTrainedTokenizerBase, samples: Sequence[Dict[str, str]], steering_hooks: Optional[List[tuple]] = None) -> Dict[str, object]:

    device = model.cfg.device
    subject_totals: Dict[str, Dict[str, int]] = {}
    correct = 0

    for example in tqdm(samples, desc="Evaluating MMLU"):
        prompt = _format_prompt(example["question"], example["choices"])
        prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)

        choice_scores = []
        for choice in example["choices"]:
            choice_ids = tokenizer(" " + choice, return_tensors="pt")["input_ids"].to(device)
            tokens = torch.cat([prompt_ids, choice_ids], dim=1)
            with torch.no_grad():
                logits = _run_with_hooks(model, tokens, steering_hooks)
            score = _choice_logprob(logits, tokens, prompt_ids.shape[1])
            choice_scores.append(score)

        pred_index = int(np.argmax(choice_scores))
        answer_value = example["answer"]
        label = answer_value if isinstance(answer_value, int) else (ord(answer_value.upper()[0]) - ord("A") if isinstance(answer_value, str) and answer_value else None)
        if label is None:
            raise ValueError(f"Unrecognized answer format: {answer_value!r}")
        
        subject = example["subject"]
        if subject not in subject_totals:
            subject_totals[subject] = {"correct": 0, "total": 0}
        subject_totals[subject]["total"] += 1
        if pred_index == label:
            subject_totals[subject]["correct"] += 1
            correct += 1

    total_questions = len(samples)
    subject_accuracies = {
        subject: stats["correct"] / stats["total"] if stats["total"] else 0.0
        for subject, stats in subject_totals.items()
    }

    return {
        "overall_accuracy": correct / total_questions if total_questions else 0.0,
        "subject_accuracies": subject_accuracies,
        "num_questions": total_questions,
    }