from __future__ import annotations

from pathlib import Path

import numpy as np

from selective_rag_rl.bandit import KnnDirectMethodBandit, MarginWeightedDirectMethodBandit, state_features
from selective_rag_rl.data import Passage, load_beir_scifact, load_hotpotqa, load_natural_questions
from selective_rag_rl.metrics import mrr, ndcg_at_k, recall_at_k
from selective_rag_rl.policy_io import load_checkpoint, save_checkpoint
from selective_rag_rl.retriever import BM25Retriever, RetrievalResult
from selective_rag_rl.rewrites import ACTIONS, rewrite, rewrite_cost


def test_hotpot_loader_reads_examples() -> None:
    path = Path("data/raw/HotpotQA/hotpot_dev_distractor_v1.json")
    examples = load_hotpotqa(path, num_examples=3, seed=42)
    assert len(examples) == 3
    assert examples[0].question
    assert examples[0].passages
    assert examples[0].gold_doc_ids


def test_bm25_retriever_is_deterministic() -> None:
    passages = [
        Passage("a", "Alpha", "Alpha apple banana"),
        Passage("b", "Beta", "Beta carrot"),
    ]
    retriever = BM25Retriever(passages)
    first = retriever.search("apple", k=2)
    second = retriever.search("apple", k=2)
    assert [r.doc_id for r in first] == [r.doc_id for r in second]
    assert first[0].doc_id == "a"


def test_rewrite_actions_are_non_empty_and_costed() -> None:
    question = "Were Scott Derrickson and Ed Wood of the same nationality?"
    for action in ACTIONS:
        output = rewrite(question, action)
        assert output.queries
        assert all(q.strip() for q in output.queries)
        assert rewrite_cost(question, output) >= 0
    assert rewrite(question, "decompose").retrieval_calls >= 1


def test_metrics_on_manual_ranking() -> None:
    results = [
        RetrievalResult("x", 3.0, 1),
        RetrievalResult("a", 2.0, 2),
        RetrievalResult("b", 1.0, 3),
    ]
    gold = {"a", "b"}
    assert recall_at_k(results, gold, 2) == 0.5
    assert mrr(results, gold) == 0.5
    assert 0.0 < ndcg_at_k(results, gold, 3) <= 1.0


def test_natural_questions_loader_builds_single_hop_pool(tmp_path: Path) -> None:
    pyarrow = __import__("pyarrow")
    parquet = __import__("pyarrow.parquet").parquet

    rows = [
        {
            "id": "q1",
            "document": {
                "title": "Ada Lovelace",
                "tokens": {
                    "token": ["Ada", "Lovelace", "wrote", "notes", "for", "the", "Analytical", "Engine"],
                    "is_html": [False] * 8,
                },
            },
            "question": {"text": "who wrote notes for the analytical engine"},
            "annotations": {
                "short_answers": [{"text": ["Ada Lovelace"]}],
                "yes_no_answer": [-1],
            },
        },
        {
            "id": "q2",
            "document": {
                "title": "Grace Hopper",
                "tokens": {
                    "token": ["Grace", "Hopper", "worked", "on", "COBOL"],
                    "is_html": [False] * 5,
                },
            },
            "question": {"text": "who worked on cobol"},
            "annotations": {
                "short_answers": [{"text": ["Grace Hopper"]}],
                "yes_no_answer": [-1],
            },
        },
        {
            "id": "q3",
            "document": {
                "title": "Alan Turing",
                "tokens": {
                    "token": ["Alan", "Turing", "studied", "computation"],
                    "is_html": [False] * 4,
                },
            },
            "question": {"text": "who studied computation"},
            "annotations": {
                "short_answers": [{"text": ["Alan Turing"]}],
                "yes_no_answer": [-1],
            },
        },
    ]
    path = tmp_path / "nq.parquet"
    parquet.write_table(pyarrow.Table.from_pylist(rows), path)

    examples = load_natural_questions(path, num_examples=2, seed=7, pool_size=3)

    assert len(examples) == 2
    assert examples[0].question
    assert examples[0].answer
    assert len(examples[0].passages) == 3
    assert examples[0].gold_doc_ids <= {p.doc_id for p in examples[0].passages}


def test_beir_scifact_loader_reads_qrels_and_samples_candidates(tmp_path: Path) -> None:
    data_dir = tmp_path / "scifact"
    qrels_dir = data_dir / "qrels"
    qrels_dir.mkdir(parents=True)
    (data_dir / "corpus.jsonl").write_text(
        "\n".join(
            [
                '{"_id": "d1", "title": "A", "text": "alpha evidence"}',
                '{"_id": "d2", "title": "B", "text": "beta evidence"}',
                '{"_id": "d3", "title": "C", "text": "negative document"}',
                '{"_id": "d4", "title": "D", "text": "another negative"}',
            ]
        ),
        encoding="utf-8",
    )
    (data_dir / "queries.jsonl").write_text(
        "\n".join(
            [
                '{"_id": "q1", "text": "alpha claim"}',
                '{"_id": "q2", "text": "beta claim"}',
            ]
        ),
        encoding="utf-8",
    )
    (qrels_dir / "test.tsv").write_text("query-id\tcorpus-id\tscore\nq1\td1\t1\nq2\td2\t1\n", encoding="utf-8")

    examples = load_beir_scifact(data_dir, num_examples=2, seed=3, split="test", pool_size=3)

    assert len(examples) == 2
    assert all(ex.qtype == "beir-scifact" for ex in examples)
    assert all(len(ex.passages) == 3 for ex in examples)
    assert all(ex.gold_doc_ids <= {p.doc_id for p in ex.passages} for ex in examples)


def test_knn_bandit_uses_nearby_rewards() -> None:
    features = np.asarray([[0.0], [0.1], [1.0], [1.1]], dtype=float)
    rewards = {
        "left": [1.0, 0.9, 0.0, 0.0],
        "right": [0.0, 0.0, 0.9, 1.0],
    }
    policy = KnnDirectMethodBandit(actions=["left", "right"], k=2)
    policy.fit(features, rewards)

    assert policy.predict(np.asarray([0.05])) == "left"
    assert policy.predict(np.asarray([1.05])) == "right"


def test_margin_weighted_bandit_downweights_near_tie_examples() -> None:
    features = np.asarray([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]], dtype=float)
    rewards = {
        "left": [1.0, 0.51, 0.50],
        "right": [0.0, 0.50, 0.49],
    }
    policy = MarginWeightedDirectMethodBandit(actions=["left", "right"], margin_floor=0.25)

    policy.fit(features, rewards)

    assert np.isclose(policy.sample_weights_[0], 1.0)
    assert policy.sample_weights_[1] < policy.sample_weights_[0]
    assert policy.sample_weights_[2] < policy.sample_weights_[0]
    assert policy.predict(np.asarray([1.0, 0.0])) == "left"


def test_state_features_adds_lexical_semantic_interactions() -> None:
    initial_results = [
        RetrievalResult("d1", 10.0, 1),
        RetrievalResult("d2", 6.0, 2),
        RetrievalResult("d3", 2.0, 3),
    ]
    semantic = [0.8, 0.7, 0.9, 0.2, 1.0, 0.05, 0.1, 0.75, 0.3]

    features = state_features("what evidence supports alpha", initial_results, semantic_features=semantic)

    assert len(features) == 29
    assert np.allclose(
        features[-6:],
        [
            features[3] * semantic[0],
            features[4] * semantic[3],
            features[5] * semantic[5],
            features[3] * semantic[8],
            features[4] * semantic[8],
            features[5] * semantic[4],
        ],
    )


def test_policy_checkpoint_round_trip(tmp_path: Path) -> None:
    features = np.asarray([[0.0], [1.0]], dtype=float)
    rewards = {"left": [1.0, 0.0], "right": [0.0, 1.0]}
    policy = KnnDirectMethodBandit(actions=["left", "right"], k=1)
    policy.fit(features, rewards)

    checkpoint = tmp_path / "policy.pkl"
    save_checkpoint(checkpoint, policy, {"dataset": "toy", "seed": 1})
    loaded = load_checkpoint(checkpoint)

    assert loaded["metadata"]["dataset"] == "toy"
    assert loaded["model"].predict(np.asarray([0.0])) == "left"
    assert loaded["model"].predict(np.asarray([1.0])) == "right"
