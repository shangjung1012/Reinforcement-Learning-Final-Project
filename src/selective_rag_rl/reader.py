from __future__ import annotations

import re
from dataclasses import dataclass

from selective_rag_rl.data import Passage
from selective_rag_rl.text import content_tokens

SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class ReaderPrediction:
    answer: str
    passage_id: str | None
    sentence: str
    score: float


class LexicalOverlapReader:
    def predict(self, question: str, passages: list[Passage]) -> ReaderPrediction:
        question_terms = set(content_tokens(question))
        best: ReaderPrediction | None = None
        for passage in passages:
            for sentence in _sentences(passage.text):
                sentence_terms = set(content_tokens(sentence))
                overlap = len(question_terms & sentence_terms)
                score = overlap / max(len(question_terms), 1)
                candidate = ReaderPrediction(
                    answer=sentence,
                    passage_id=passage.doc_id,
                    sentence=sentence,
                    score=float(score),
                )
                if best is None or (candidate.score, candidate.answer) > (best.score, best.answer):
                    best = candidate
        return best or ReaderPrediction(answer="", passage_id=None, sentence="", score=0.0)


def _sentences(text: str) -> list[str]:
    sentences = [part.strip() for part in SENTENCE_RE.split(text) if part.strip()]
    return sentences or ([text.strip()] if text.strip() else [])
