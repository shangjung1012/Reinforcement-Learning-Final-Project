from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

from selective_rag_rl.core.data import load_hotpotqa, split_examples
from selective_rag_rl.experiments.gemini_baseline import (
    GEMINI_MODES,
    GeminiBudgetError,
    GeminiCache,
    RewriteProvider,
    evaluate_gemini_rewrites,
)


def run_repeated_gemini_baseline(
    *,
    data_path: Path,
    output_dir: Path,
    seeds: list[int],
    num_examples: int,
    cache_path: Path,
    k: int = 5,
    rewrite_provider: RewriteProvider | None = None,
    allow_api: bool = False,
    max_new_calls: int = 0,
    dry_run: bool = False,
) -> dict[str, object]:
    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    estimated_new_calls = estimate_unique_cache_misses(
        data_path=data_path,
        seeds=seeds,
        num_examples=num_examples,
        cache_path=cache_path,
    )
    preflight_rows = [
        _seed_preflight_row(
            data_path=data_path,
            seed=seed,
            num_examples=num_examples,
            cache_path=cache_path,
        )
        for seed in seeds
    ]
    seed_preflight_csv = results_dir / "gemini_repeated_seed_preflight.csv"
    pd.DataFrame(preflight_rows).to_csv(seed_preflight_csv, index=False)

    metadata: dict[str, object] = {
        "dataset": "HotpotQA repeated Gemini rewrite baselines",
        "seeds": seeds,
        "num_examples": num_examples,
        "k": k,
        "cache": str(cache_path),
        "estimated_new_calls": estimated_new_calls,
        "allow_api": allow_api,
        "max_new_calls": max_new_calls,
        "dry_run": dry_run,
        "outputs": {
            "metadata_json": str(results_dir / "gemini_repeated_metadata.json"),
            "seed_preflight_csv": str(seed_preflight_csv),
        },
    }
    if dry_run:
        Path(metadata["outputs"]["metadata_json"]).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return metadata

    if estimated_new_calls > max_new_calls:
        raise GeminiBudgetError(
            f"Estimated Gemini cache misses ({estimated_new_calls}) exceed max_new_calls ({max_new_calls})"
        )
    if estimated_new_calls > 0 and rewrite_provider is None and not allow_api:
        raise GeminiBudgetError("Gemini cache misses require allow_api=True before live API calls")

    seed_frames = []
    for seed in seeds:
        run_dir = output_dir / "runs" / f"seed_{seed}"
        run_metadata = evaluate_gemini_rewrites(
            data_path=data_path,
            output_dir=run_dir,
            num_examples=num_examples,
            seed=seed,
            k=k,
            cache_path=cache_path,
            rewrite_provider=rewrite_provider,
            allow_api=allow_api,
            max_new_calls=max_new_calls,
        )
        frame = pd.read_csv(run_metadata["outputs"]["summary_csv"])
        frame["seed"] = seed
        frame["test_examples"] = run_metadata["test_examples"]
        frame["cache_hits"] = run_metadata["cache_hits"]
        frame["cache_misses"] = run_metadata["cache_misses"]
        seed_frames.append(frame)

    seed_summary = pd.concat(seed_frames, ignore_index=True)
    aggregate = _aggregate_seed_summary(seed_summary)
    seed_summary_csv = results_dir / "gemini_repeated_seed_summary.csv"
    summary_csv = results_dir / "gemini_repeated_summary.csv"
    summary_json = results_dir / "gemini_repeated_summary.json"
    seed_summary.to_csv(seed_summary_csv, index=False)
    aggregate.to_csv(summary_csv, index=False)
    summary_json.write_text(json.dumps(aggregate.to_dict(orient="records"), indent=2), encoding="utf-8")
    metadata["outputs"].update(
        {
            "seed_summary_csv": str(seed_summary_csv),
            "summary_csv": str(summary_csv),
            "summary_json": str(summary_json),
        }
    )
    Path(metadata["outputs"]["metadata_json"]).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def estimate_unique_cache_misses(
    *,
    data_path: Path,
    seeds: list[int],
    num_examples: int,
    cache_path: Path,
) -> int:
    cache = GeminiCache(cache_path)
    required = set()
    for seed in seeds:
        examples = load_hotpotqa(data_path, num_examples=num_examples, seed=seed)
        _train, test = split_examples(examples)
        for ex in test:
            for mode, _method in GEMINI_MODES:
                if cache.get(ex.qid, mode) is None:
                    required.add((ex.qid, mode))
    return len(required)


def _seed_preflight_row(*, data_path: Path, seed: int, num_examples: int, cache_path: Path) -> dict[str, object]:
    cache = GeminiCache(cache_path)
    examples = load_hotpotqa(data_path, num_examples=num_examples, seed=seed)
    _train, test = split_examples(examples)
    misses = 0
    hits = 0
    for ex in test:
        for mode, _method in GEMINI_MODES:
            if cache.get(ex.qid, mode) is None:
                misses += 1
            else:
                hits += 1
    return {
        "seed": seed,
        "num_examples": len(examples),
        "test_examples": len(test),
        "cache_hits": hits,
        "cache_misses": misses,
    }


def _aggregate_seed_summary(seed_summary: pd.DataFrame) -> pd.DataFrame:
    metric_columns = ["recall_at_5", "mrr", "ndcg_at_5", "reward", "rewrite_cost", "retrieval_calls"]
    rows = []
    for method, group in seed_summary.groupby("method", sort=False):
        row: dict[str, object] = {
            "method": method,
            "seeds": int(group["seed"].nunique()),
            "total_test_examples": int(group["test_examples"].sum()),
            "total_cache_hits": int(group["cache_hits"].sum()),
            "total_cache_misses": int(group["cache_misses"].sum()),
        }
        for metric in metric_columns:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=0)) if len(group) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run bounded repeated-seed Gemini rewrite baselines.",
        allow_abbrev=False,
    )
    parser.add_argument("--data-path", type=Path, default=Path("data/raw/HotpotQA/hotpot_dev_distractor_v1.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/codex_gemini_repeated"))
    parser.add_argument("--seeds", default="41,42,43")
    parser.add_argument("--num-examples", type=int, default=10)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--cache-path", type=Path, default=Path("outputs/cache/codex_gemini_repeated.jsonl"))
    parser.add_argument("--allow-api", action="store_true")
    parser.add_argument("--max-new-calls", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    allow_api = args.allow_api or os.environ.get("CODEX_ALLOW_API_CALLS") == "1"
    seeds = [int(seed.strip()) for seed in args.seeds.split(",") if seed.strip()]
    metadata = run_repeated_gemini_baseline(
        data_path=args.data_path,
        output_dir=args.output_dir,
        seeds=seeds,
        num_examples=args.num_examples,
        k=args.k,
        cache_path=args.cache_path,
        allow_api=allow_api,
        max_new_calls=args.max_new_calls,
        dry_run=args.dry_run,
    )
    print(json.dumps(metadata, indent=2))
    summary_csv = metadata["outputs"].get("summary_csv")
    if summary_csv:
        print(pd.read_csv(summary_csv).to_string(index=False))


if __name__ == "__main__":
    main()
