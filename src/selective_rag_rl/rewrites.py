from __future__ import annotations

from dataclasses import dataclass

from selective_rag_rl.text import capitalized_spans, content_tokens, unique_preserve_order

ACTIONS = ["keep", "keyword_compress", "entity_expand", "decompose"]


@dataclass(frozen=True)
class RewriteOutput:
    action: str
    queries: list[str]

    @property
    def retrieval_calls(self) -> int:
        return len(self.queries)

    @property
    def joined_query(self) -> str:
        return " ".join(self.queries)


def rewrite(question: str, action: str) -> RewriteOutput:
    if action == "keep":
        return RewriteOutput(action=action, queries=[question.strip()])
    if action == "keyword_compress":
        query = " ".join(content_tokens(question)) or question
        return RewriteOutput(action=action, queries=[query])
    if action == "entity_expand":
        spans = capitalized_spans(question)
        content = content_tokens(question)
        query = " ".join(unique_preserve_order([*spans, *content])) or question
        return RewriteOutput(action=action, queries=[query])
    if action == "decompose":
        return RewriteOutput(action=action, queries=_decompose(question))
    raise ValueError(f"Unknown rewrite action: {action}")


def rewrite_cost(question: str, output: RewriteOutput) -> float:
    original_len = max(1, len(content_tokens(question)))
    rewrite_len = sum(len(content_tokens(q)) for q in output.queries)
    extra_tokens = max(0, rewrite_len - original_len)
    extra_calls = max(0, output.retrieval_calls - 1)
    return 0.02 * extra_tokens + 0.10 * extra_calls


def _decompose(question: str) -> list[str]:
    cleaned = question.strip()
    separators = [" and ", " or ", " after ", " before ", " while ", " when "]
    lower = cleaned.lower()
    for sep in separators:
        if sep in lower:
            idx = lower.index(sep)
            left = cleaned[:idx].strip(" ,?")
            right = cleaned[idx + len(sep) :].strip(" ,?")
            queries = [q for q in [left, right] if q]
            if len(queries) == 2:
                return queries

    spans = capitalized_spans(cleaned)
    if len(spans) >= 2:
        return [f"{spans[0]} {cleaned}", f"{spans[1]} {cleaned}"]
    tokens = content_tokens(cleaned)
    midpoint = max(1, len(tokens) // 2)
    q1 = " ".join(tokens[:midpoint])
    q2 = " ".join(tokens[midpoint:])
    return [q for q in [q1, q2] if q] or [cleaned]

