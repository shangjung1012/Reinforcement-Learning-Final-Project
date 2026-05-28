from __future__ import annotations

from selective_rag_rl.data import Passage
from selective_rag_rl.reader import LexicalOverlapReader


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
