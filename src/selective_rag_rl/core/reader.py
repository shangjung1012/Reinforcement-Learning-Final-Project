from __future__ import annotations

import re
from dataclasses import dataclass

from selective_rag_rl.core.data import Passage
from selective_rag_rl.core.text import content_tokens

SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
MONTH_RE = re.compile(
    r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)"
    r"\s+\d{1,2},\s+\d{4}\b"
)
YEAR_RE = re.compile(r"\b(?:1[5-9]\d{2}|20\d{2})\b")
DATE_QUESTION_RE = re.compile(r"\b(?:when|date|year)\b", flags=re.IGNORECASE)
YES_NO_QUESTION_RE = re.compile(r"^\s*(?:are|is|was|were|do|does|did|has|have|had|can|could|will|would)\b", flags=re.IGNORECASE)
YES_NO_ANSWER_RE = re.compile(r"\b(?:yes|no)\b", flags=re.IGNORECASE)
NUMBER_QUESTION_RE = re.compile(r"\b(?:how many|how much|number|count)\b", flags=re.IGNORECASE)
NUMBER_RE = re.compile(r"\b\d+(?:,\d{3})*(?:\.\d+)?\b")
CAPITALIZED_SPAN_RE = re.compile(
    r"\b[A-Z][A-Za-z0-9.'-]*(?:\s+[A-Z][A-Za-z0-9.'-]*){0,5}\b"
)


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


class SpanHeuristicReader:
    def predict(self, question: str, passages: list[Passage]) -> ReaderPrediction:
        sentence_prediction = LexicalOverlapReader().predict(question, passages)
        if not sentence_prediction.answer:
            return sentence_prediction
        answer = _extract_answer_span(question, sentence_prediction.sentence)
        return ReaderPrediction(
            answer=answer,
            passage_id=sentence_prediction.passage_id,
            sentence=sentence_prediction.sentence,
            score=sentence_prediction.score,
        )


class AnswerTypeHeuristicReader:
    def predict(self, question: str, passages: list[Passage]) -> ReaderPrediction:
        sentence_prediction = LexicalOverlapReader().predict(question, passages)
        if not sentence_prediction.answer:
            return sentence_prediction
        answer = _extract_answer_by_type(question, sentence_prediction.sentence)
        return ReaderPrediction(
            answer=answer,
            passage_id=sentence_prediction.passage_id,
            sentence=sentence_prediction.sentence,
            score=sentence_prediction.score,
        )


def _sentences(text: str) -> list[str]:
    sentences = [part.strip() for part in SENTENCE_RE.split(text) if part.strip()]
    return sentences or ([text.strip()] if text.strip() else [])


def _extract_answer_span(question: str, sentence: str) -> str:
    question_terms = set(content_tokens(question))
    if DATE_QUESTION_RE.search(question):
        date = MONTH_RE.search(sentence)
        if date:
            return date.group(0)
        year = YEAR_RE.search(sentence)
        if year:
            return year.group(0)

    capitalized = _capitalized_candidates(sentence, question_terms)
    if capitalized:
        return capitalized[0]
    return sentence.strip()


def _extract_answer_by_type(question: str, sentence: str) -> str:
    if YES_NO_QUESTION_RE.search(question):
        yes_no = YES_NO_ANSWER_RE.search(sentence)
        if yes_no:
            return yes_no.group(0).lower()
    if NUMBER_QUESTION_RE.search(question):
        number = NUMBER_RE.search(sentence)
        if number:
            return number.group(0)
    return _extract_answer_span(question, sentence)


def _capitalized_candidates(sentence: str, question_terms: set[str]) -> list[str]:
    candidates = []
    for match in CAPITALIZED_SPAN_RE.finditer(sentence):
        candidate = match.group(0).strip()
        candidate_terms = set(content_tokens(candidate))
        if not candidate_terms:
            continue
        novelty = len(candidate_terms - question_terms)
        if novelty == 0:
            continue
        candidates.append((novelty, len(candidate_terms), match.start(), candidate))
    candidates.sort(key=lambda item: (-item[0], -item[1], item[2]))
    return [candidate for *_rest, candidate in candidates]
