from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

from selective_rag_rl.preflight.api_preflight import run_api_preflight


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a sanitized Google GenAI/Vertex API preflight.")
    parser.add_argument("--provider", choices=["gemini", "vertex-embedding", "all"], default="all")
    parser.add_argument("--cache-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/codex_api_preflight"))
    parser.add_argument("--allow-api", action="store_true")
    parser.add_argument("--max-new-gemini-calls", type=int, default=0)
    parser.add_argument("--max-new-embedding-texts", type=int, default=0)
    parser.add_argument("--sample-text", action="append", default=None)
    args = parser.parse_args()

    allow_api = args.allow_api or os.environ.get("CODEX_ALLOW_API_CALLS") == "1"
    summary = run_api_preflight(
        project_root=Path.cwd(),
        output_dir=args.output_dir,
        provider=args.provider,
        cache_path=args.cache_path,
        allow_api=allow_api,
        max_new_gemini_calls=args.max_new_gemini_calls,
        max_new_embedding_texts=args.max_new_embedding_texts,
        sample_texts=args.sample_text,
    )
    print(json.dumps(_printable_summary(summary), indent=2))
    csv_path = Path(summary["outputs"]["summary_csv"])
    if csv_path.exists():
        print(pd.read_csv(csv_path).to_string(index=False))


def _printable_summary(summary: dict[str, object]) -> dict[str, object]:
    return {
        "environment": summary["environment"],
        "cache": summary["cache"],
        "allow_api": summary["allow_api"],
        "providers": summary["providers"],
        "outputs": summary["outputs"],
    }


if __name__ == "__main__":
    main()
