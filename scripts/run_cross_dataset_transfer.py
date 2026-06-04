from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from selective_rag_rl.experiments.cross_dataset_transfer import run_beir_transfer_matrix


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scifact-path", type=Path, default=Path("data/raw/scifact"))
    parser.add_argument("--nfcorpus-path", type=Path, default=Path("data/raw/nfcorpus"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--num-train-examples", type=int, default=600)
    parser.add_argument("--num-test-examples", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--full-corpus", action="store_true")
    parser.add_argument("--embedder", default="sentence-transformers/all-MiniLM-L6-v2")
    args = parser.parse_args()

    metadata = run_beir_transfer_matrix(
        scifact_path=args.scifact_path,
        nfcorpus_path=args.nfcorpus_path,
        output_dir=args.output_dir,
        num_train_examples=args.num_train_examples,
        num_test_examples=args.num_test_examples,
        seed=args.seed,
        full_corpus=args.full_corpus,
        embedder_name=args.embedder,
    )
    print(pd.read_csv(metadata["summary_csv"]).to_string(index=False))


if __name__ == "__main__":
    main()
