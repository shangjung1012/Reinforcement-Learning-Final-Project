from __future__ import annotations

from selective_rag_rl.core.answer_metrics import exact_match, normalize_answer, token_f1


def test_normalize_answer_uses_squad_style_rules() -> None:
    assert normalize_answer("The, QUICK!! Brown   fox.") == "quick brown fox"
    assert normalize_answer("An answer") == "answer"


def test_exact_match_compares_normalized_strings() -> None:
    assert exact_match("The Eiffel Tower!", "eiffel tower") == 1.0
    assert exact_match("Eiffel Tower", "Louvre Museum") == 0.0


def test_token_f1_scores_overlap_and_empty_answers() -> None:
    assert token_f1("Ada Lovelace wrote notes", "Ada Lovelace") == 2 * 2 / (4 + 2)
    assert token_f1("", "") == 1.0
    assert token_f1("", "Ada") == 0.0
    assert token_f1("Ada", "") == 0.0
