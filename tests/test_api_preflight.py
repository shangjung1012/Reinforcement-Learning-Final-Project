from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from selective_rag_rl.preflight.api_preflight import run_api_preflight


def test_api_preflight_dry_run_never_calls_providers_and_hides_values(tmp_path: Path, monkeypatch) -> None:
    credentials = tmp_path / "application_default_credentials.json"
    credentials.write_text("{}", encoding="utf-8")
    env_path = tmp_path / ".env"
    env_path.write_text(
        "\n".join(
            [
                "GOOGLE_GENAI_USE_VERTEXAI=true",
                "GOOGLE_CLOUD_PROJECT=secret-project",
                "GOOGLE_CLOUD_LOCATION=global",
                f"GOOGLE_APPLICATION_CREDENTIALS={credentials.as_posix()}",
                "GEMINI_MODEL=gemini-2.5-pro",
            ]
        ),
        encoding="utf-8",
    )
    calls: list[str] = []

    summary = run_api_preflight(
        project_root=tmp_path,
        output_dir=tmp_path / "out",
        provider="all",
        allow_api=False,
        max_new_gemini_calls=1,
        max_new_embedding_texts=1,
        gemini_probe=lambda: calls.append("gemini") or "ok",
        embedding_probe=lambda texts: calls.append("embedding") or len(texts),
    )

    assert calls == []
    assert summary["environment"]["env_file_exists"] is True
    assert summary["environment"]["required_variables"]["GOOGLE_CLOUD_PROJECT"] == "present"
    assert summary["environment"]["required_variables"]["GOOGLE_APPLICATION_CREDENTIALS"] == "present"
    assert summary["environment"]["credential_file_exists"] is True
    assert summary["environment"]["credential_file_basename"] == "application_default_credentials.json"
    serialized = json.dumps(summary)
    assert "secret-project" not in serialized
    assert str(credentials) not in serialized

    rows = pd.read_csv(tmp_path / "out" / "api_preflight_summary.csv")
    assert set(rows["provider"]) == {"gemini", "vertex-embedding"}
    assert rows["actual_calls_or_texts"].sum() == 0
    assert set(rows["result"]) == {"dry_run_no_api_call"}


def test_api_preflight_allow_api_respects_one_call_budgets(tmp_path: Path) -> None:
    credentials = tmp_path / "adc.json"
    credentials.write_text("{}", encoding="utf-8")
    (tmp_path / ".env").write_text(
        "\n".join(
            [
                "GOOGLE_CLOUD_PROJECT=project",
                f"GOOGLE_APPLICATION_CREDENTIALS={credentials.as_posix()}",
            ]
        ),
        encoding="utf-8",
    )
    calls: list[str] = []

    summary = run_api_preflight(
        project_root=tmp_path,
        output_dir=tmp_path / "out",
        provider="all",
        allow_api=True,
        max_new_gemini_calls=1,
        max_new_embedding_texts=1,
        sample_texts=["harmless embedding text"],
        gemini_probe=lambda: calls.append("gemini") or "rewritten query",
        embedding_probe=lambda texts: calls.append(f"embedding:{len(texts)}") or len(texts),
    )

    assert calls == ["gemini", "embedding:1"]
    rows = pd.DataFrame(summary["providers"])
    assert set(rows["result"]) == {"api_call_succeeded"}
    assert rows["actual_calls_or_texts"].sum() == 2


def test_api_preflight_blocks_api_when_budget_is_zero(tmp_path: Path) -> None:
    credentials = tmp_path / "adc.json"
    credentials.write_text("{}", encoding="utf-8")
    (tmp_path / ".env").write_text(
        "\n".join(
            [
                "GOOGLE_CLOUD_PROJECT=project",
                f"GOOGLE_APPLICATION_CREDENTIALS={credentials.as_posix()}",
            ]
        ),
        encoding="utf-8",
    )
    calls: list[str] = []

    summary = run_api_preflight(
        project_root=tmp_path,
        output_dir=tmp_path / "out",
        provider="gemini",
        allow_api=True,
        max_new_gemini_calls=0,
        gemini_probe=lambda: calls.append("gemini") or "ok",
    )

    assert calls == []
    row = summary["providers"][0]
    assert row["provider"] == "gemini"
    assert row["result"] == "blocked_budget_exceeded"
    assert row["actual_calls_or_texts"] == 0
