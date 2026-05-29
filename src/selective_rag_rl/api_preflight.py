from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import pandas as pd
from dotenv import dotenv_values, load_dotenv

ProviderChoice = Literal["gemini", "vertex-embedding", "all"]
GeminiProbe = Callable[[], str]
EmbeddingProbe = Callable[[list[str]], int]

REQUIRED_VARIABLES = ("GOOGLE_CLOUD_PROJECT", "GOOGLE_APPLICATION_CREDENTIALS")
OPTIONAL_VARIABLES = ("GOOGLE_GENAI_USE_VERTEXAI", "GOOGLE_CLOUD_LOCATION", "GEMINI_MODEL")
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
DEFAULT_LOCATION = "us-central1"
DEFAULT_GEMINI_SAMPLE = "Which author lived longer, Nelson Algren or Nathanael West?"
DEFAULT_EMBEDDING_SAMPLE = "A harmless API preflight sentence for retrieval embedding."


def run_api_preflight(
    project_root: Path,
    output_dir: Path,
    provider: ProviderChoice = "all",
    cache_path: Path | None = None,
    allow_api: bool = False,
    max_new_gemini_calls: int = 0,
    max_new_embedding_texts: int = 0,
    sample_texts: list[str] | None = None,
    gemini_probe: GeminiProbe | None = None,
    embedding_probe: EmbeddingProbe | None = None,
) -> dict[str, object]:
    """Run a sanitized API preflight and optionally one tiny call per provider."""

    project_root = project_root.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    env_path = project_root / ".env"
    env_values = _merged_env(env_path)
    environment = _environment_summary(project_root, env_path, env_values)
    cache_summary = _cache_summary(cache_path)
    provider_names = _provider_names(provider)

    rows = [
        _provider_preflight(
            provider_name=name,
            project_root=project_root,
            env_values=env_values,
            environment_ok=_environment_ok(environment),
            allow_api=allow_api,
            max_new_gemini_calls=max_new_gemini_calls,
            max_new_embedding_texts=max_new_embedding_texts,
            sample_texts=sample_texts,
            gemini_probe=gemini_probe,
            embedding_probe=embedding_probe,
        )
        for name in provider_names
    ]
    summary = {
        "environment": environment,
        "cache": cache_summary,
        "allow_api": bool(allow_api),
        "providers": rows,
        "outputs": {
            "summary_json": str(output_dir / "api_preflight_summary.json"),
            "summary_csv": str(output_dir / "api_preflight_summary.csv"),
        },
    }
    (output_dir / "api_preflight_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_csv(output_dir / "api_preflight_summary.csv", index=False)
    return summary


def _merged_env(env_path: Path) -> dict[str, str]:
    file_values = dotenv_values(env_path) if env_path.exists() else {}
    merged: dict[str, str] = {}
    for key in (*REQUIRED_VARIABLES, *OPTIONAL_VARIABLES):
        value = file_values.get(key) or os.environ.get(key)
        if value is not None:
            merged[key] = str(value)
    return merged


def _environment_summary(project_root: Path, env_path: Path, env_values: dict[str, str]) -> dict[str, object]:
    credentials_value = env_values.get("GOOGLE_APPLICATION_CREDENTIALS")
    credential_path = _resolve_path(project_root, credentials_value) if credentials_value else None
    return {
        "env_file_exists": env_path.exists(),
        "required_variables": {key: _presence(env_values.get(key)) for key in REQUIRED_VARIABLES},
        "optional_variables": {
            "GOOGLE_GENAI_USE_VERTEXAI": _presence(env_values.get("GOOGLE_GENAI_USE_VERTEXAI")),
            "GOOGLE_CLOUD_LOCATION": _presence(env_values.get("GOOGLE_CLOUD_LOCATION"), default_label="missing_uses_default"),
            "GEMINI_MODEL": _presence(env_values.get("GEMINI_MODEL"), default_label="missing_uses_default"),
        },
        "credential_file_exists": bool(credential_path and credential_path.exists()),
        "credential_file_basename": credential_path.name if credential_path else None,
    }


def _presence(value: str | None, default_label: str = "missing") -> str:
    return "present" if value else default_label


def _resolve_path(project_root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else project_root / path


def _environment_ok(environment: dict[str, object]) -> bool:
    required = environment["required_variables"]
    assert isinstance(required, dict)
    return all(required.get(key) == "present" for key in REQUIRED_VARIABLES) and bool(environment["credential_file_exists"])


def _cache_summary(cache_path: Path | None) -> dict[str, object]:
    if cache_path is None:
        return {"cache_path": None, "cache_exists": False, "cache_rows": 0}
    row_count = 0
    if cache_path.exists():
        row_count = sum(1 for line in cache_path.read_text(encoding="utf-8").splitlines() if line.strip())
    return {"cache_path": str(cache_path), "cache_exists": cache_path.exists(), "cache_rows": row_count}


def _provider_names(provider: ProviderChoice) -> list[str]:
    if provider == "all":
        return ["gemini", "vertex-embedding"]
    if provider in {"gemini", "vertex-embedding"}:
        return [provider]
    raise ValueError("provider must be one of: gemini, vertex-embedding, all")


def _provider_preflight(
    provider_name: str,
    project_root: Path,
    env_values: dict[str, str],
    environment_ok: bool,
    allow_api: bool,
    max_new_gemini_calls: int,
    max_new_embedding_texts: int,
    sample_texts: list[str] | None,
    gemini_probe: GeminiProbe | None,
    embedding_probe: EmbeddingProbe | None,
) -> dict[str, object]:
    allowed = max_new_gemini_calls if provider_name == "gemini" else max_new_embedding_texts
    estimated_misses = 1
    base = {
        "provider": provider_name,
        "mode": "preflight",
        "allow_api": bool(allow_api),
        "new_calls_or_texts_allowed": int(allowed),
        "estimated_misses": estimated_misses,
        "actual_calls_or_texts": 0,
        "result": "",
        "error_type": "",
        "error_message": "",
    }
    if not environment_ok:
        return {**base, "result": "blocked_missing_credentials"}
    if not allow_api:
        return {**base, "result": "dry_run_no_api_call"}
    if allowed < estimated_misses:
        return {**base, "result": "blocked_budget_exceeded"}

    try:
        if provider_name == "gemini":
            probe = gemini_probe or _default_gemini_probe(project_root, env_values)
            probe()
            return {**base, "actual_calls_or_texts": 1, "result": "api_call_succeeded"}
        texts = sample_texts or [DEFAULT_EMBEDDING_SAMPLE]
        texts = texts[:1]
        probe = embedding_probe or _default_embedding_probe(project_root, env_values)
        actual = int(probe(texts))
        return {**base, "actual_calls_or_texts": actual, "result": "api_call_succeeded"}
    except Exception as exc:  # pragma: no cover - exercised manually against live APIs
        error_type, error_message = _sanitize_error(exc, env_values, project_root)
        return {**base, "result": "api_call_failed", "error_type": error_type, "error_message": error_message}


def _default_gemini_probe(project_root: Path, env_values: dict[str, str]) -> GeminiProbe:
    def probe() -> str:
        from google import genai
        from google.genai import types

        _prepare_live_env(project_root)
        client = genai.Client(
            vertexai=True,
            project=env_values["GOOGLE_CLOUD_PROJECT"],
            location=env_values.get("GOOGLE_CLOUD_LOCATION", DEFAULT_LOCATION),
        )
        response = client.models.generate_content(
            model=env_values.get("GEMINI_MODEL", DEFAULT_GEMINI_MODEL),
            contents=(
                "Rewrite this question into one concise search query. "
                "Return only the query.\n\n"
                f"Question: {DEFAULT_GEMINI_SAMPLE}"
            ),
            config=types.GenerateContentConfig(temperature=0.0, max_output_tokens=32),
        )
        return response.text or ""

    return probe


def _default_embedding_probe(project_root: Path, env_values: dict[str, str]) -> EmbeddingProbe:
    def probe(texts: list[str]) -> int:
        from google import genai
        from google.genai import types

        _prepare_live_env(project_root)
        client = genai.Client(
            vertexai=True,
            project=env_values["GOOGLE_CLOUD_PROJECT"],
            location=env_values.get("GOOGLE_CLOUD_LOCATION", DEFAULT_LOCATION),
        )
        response = client.models.embed_content(
            model="gemini-embedding-001",
            contents=texts,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
        )
        return len(response.embeddings or [])

    return probe


def _prepare_live_env(project_root: Path) -> None:
    env_path = project_root / ".env"
    load_dotenv(env_path)
    credentials = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if credentials:
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(_resolve_path(project_root, credentials))


def _sanitize_error(exc: Exception, env_values: dict[str, str], project_root: Path) -> tuple[str, str]:
    message = str(exc)
    for value in env_values.values():
        if value:
            message = message.replace(value, "<redacted>")
    credentials = env_values.get("GOOGLE_APPLICATION_CREDENTIALS")
    if credentials:
        message = message.replace(str(_resolve_path(project_root, credentials)), "<redacted>")
    return exc.__class__.__name__, message[:300]
