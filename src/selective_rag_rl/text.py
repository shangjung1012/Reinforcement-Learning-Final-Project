from __future__ import annotations

import math
import re
from collections.abc import Iterable

TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
CAPITALIZED_RE = re.compile(r"\b(?:[A-Z][a-z0-9]+(?:\s+[A-Z][a-z0-9]+)*)\b")

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "did",
    "do",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "he",
    "her",
    "his",
    "how",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "she",
    "that",
    "the",
    "their",
    "this",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "whom",
    "whose",
    "why",
    "with",
}

WH_WORDS = ["who", "what", "when", "where", "which", "why", "how", "yesno"]


def tokenize(text: str) -> list[str]:
    return [m.group(0).lower() for m in TOKEN_RE.finditer(text)]


def content_tokens(text: str) -> list[str]:
    return [tok for tok in tokenize(text) if tok not in STOPWORDS]


def unique_preserve_order(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        key = item.lower()
        if key and key not in seen:
            seen.add(key)
            out.append(item)
    return out


def capitalized_spans(text: str) -> list[str]:
    spans = [m.group(0).strip() for m in CAPITALIZED_RE.finditer(text)]
    return unique_preserve_order([s for s in spans if s.lower() not in STOPWORDS])


def wh_word(text: str) -> str:
    toks = tokenize(text)
    if toks and toks[0] in {"was", "were", "is", "are", "did", "does", "do", "can"}:
        return "yesno"
    for word in WH_WORDS[:-1]:
        if word in toks:
            return word
    return "what"


def entropy(scores: list[float]) -> float:
    if not scores:
        return 0.0
    shifted = [s - max(scores) for s in scores]
    exps = [math.exp(s) for s in shifted]
    denom = sum(exps)
    if denom <= 0:
        return 0.0
    probs = [v / denom for v in exps]
    return -sum(p * math.log(p + 1e-12) for p in probs)

