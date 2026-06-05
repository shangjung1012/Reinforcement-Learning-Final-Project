from __future__ import annotations

from selective_rag_rl.core.data import Passage
from selective_rag_rl.core.reader import LexicalOverlapReader, SpanHeuristicReader


def test_lexical_overlap_reader_selects_sentence_with_question_terms() -> None:
    reader = LexicalOverlapReader()
    passages = [
        Passage("noise", "Noise", "Noise. This sentence is unrelated."),
        Passage(
            "ada",
            "Ada Lovelace",
            "Ada Lovelace. Ada Lovelace wrote notes for the Analytical Engine.",
        ),
    ]

    prediction = reader.predict("Who wrote notes for the Analytical Engine?", passages)

    assert prediction.passage_id == "ada"
    assert "Ada Lovelace" in prediction.answer
    assert prediction.score > 0


def test_lexical_overlap_reader_handles_empty_passages() -> None:
    prediction = LexicalOverlapReader().predict("Who wrote notes?", [])

    assert prediction.answer == ""
    assert prediction.passage_id is None
    assert prediction.score == 0.0


def test_span_heuristic_reader_extracts_capitalized_answer_span() -> None:
    reader = SpanHeuristicReader()
    passages = [
        Passage(
            "ada",
            "Ada Lovelace",
            "Ada Lovelace. Ada Lovelace wrote notes for the Analytical Engine.",
        ),
    ]

    prediction = reader.predict("Who wrote notes for the Analytical Engine?", passages)

    assert prediction.passage_id == "ada"
    assert prediction.answer == "Ada Lovelace"
    assert prediction.sentence == "Ada Lovelace wrote notes for the Analytical Engine."
    assert prediction.score > 0


def test_span_heuristic_reader_extracts_date_answer_span() -> None:
    reader = SpanHeuristicReader()
    passages = [
        Passage(
            "moon",
            "Moon landing",
            "Apollo 11 landed on the Moon on July 20, 1969.",
        ),
    ]

    prediction = reader.predict("When did Apollo 11 land on the Moon?", passages)

    assert prediction.answer == "July 20, 1969"


def test_span_heuristic_reader_falls_back_to_sentence_when_no_span_matches() -> None:
    reader = SpanHeuristicReader()
    passages = [Passage("plain", "Plain", "the answer is hidden in lowercase words only.")]

    prediction = reader.predict("Where is the answer hidden?", passages)

    assert prediction.answer == "the answer is hidden in lowercase words only."
