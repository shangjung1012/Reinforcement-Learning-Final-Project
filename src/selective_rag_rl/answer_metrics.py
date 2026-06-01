from __future__ import annotations

import re
import string
from collections import Counter

ARTICLES_RE = re.compile(r"\b(a|an|the)\b", flags=re.IGNORECASE)


def normalize_answer(text: str) -> str:
    """SQuAD-style answer normalization."""
    lowered = text.lower()
    no_punctuation = lowered.translate(str.maketrans("", "", string.punctuation))
    no_articles = ARTICLES_RE.sub(" ", no_punctuation)
    return " ".join(no_articles.split())


def exact_match(prediction: str, gold: str) -> float:
    return 1.0 if normalize_answer(prediction) == normalize_answer(gold) else 0.0


def token_f1(prediction: str, gold: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens or not gold_tokens:
        return 1.0 if pred_tokens == gold_tokens else 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)
