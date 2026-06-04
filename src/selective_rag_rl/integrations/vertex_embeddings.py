from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

import numpy as np
from dotenv import load_dotenv

EmbeddingFetcher = Callable[[list[str], str], list[np.ndarray]]


class SemanticEmbedder(Protocol):
    def embed_text(self, text: str, task_type: str) -> np.ndarray:
        ...

    def embed_texts(self, texts: list[str], task_type: str) -> list[np.ndarray]:
        ...


class VertexEmbeddingBudgetError(RuntimeError):
    def __init__(self, message: str, *, missing: int, allowed: int, dry_run: bool) -> None:
        super().__init__(message)
        self.missing = int(missing)
        self.allowed = int(allowed)
        self.dry_run = bool(dry_run)


class VertexTextEmbeddingProvider:
    def __init__(
        self,
        project_root: Path,
        cache_path: Path,
        model: str = "gemini-embedding-001",
        batch_size: int = 32,
        allow_api: bool = False,
        max_new_texts: int = 0,
        dry_run: bool = False,
        fetcher: EmbeddingFetcher | None = None,
    ) -> None:
        self.project_root = project_root
        self.model = model
        self.batch_size = batch_size
        self.allow_api = allow_api
        self.max_new_texts = int(max_new_texts)
        self.dry_run = dry_run
        self._fetcher = fetcher
        self._client = None
        self.cache_hits = 0
        self.cache_misses = 0
        self.actual_new_texts = 0
        self.cache_path = cache_path
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache: dict[tuple[str, str, str], np.ndarray] = {}
        self._load_cache()

    def embed_text(self, text: str, task_type: str) -> np.ndarray:
        return self.embed_texts([text], task_type=task_type)[0]

    def embed_texts(self, texts: list[str], task_type: str) -> list[np.ndarray]:
        normalized_texts = [text.strip() for text in texts]
        missing = [text for text in dict.fromkeys(normalized_texts) if self._key(text, task_type) not in self._cache]
        self.cache_misses += len(missing)
        self.cache_hits += len(normalized_texts) - len(missing)
        self._check_budget(len(missing))
        for start in range(0, len(missing), self.batch_size):
            batch = missing[start : start + self.batch_size]
            vectors = self._fetch(batch, task_type)
            for text, vector in zip(batch, vectors, strict=True):
                key = self._key(text, task_type)
                self._cache[key] = vector
                self._append_cache(key, vector)
                self.actual_new_texts += 1
        return [self._cache[self._key(text, task_type)] for text in normalized_texts]

    def _fetch(self, texts: list[str], task_type: str) -> list[np.ndarray]:
        if self._fetcher is not None:
            return self._fetcher(texts, task_type)
        from google.genai import types

        response = self._live_client().models.embed_content(
            model=self.model,
            contents=texts,
            config=types.EmbedContentConfig(task_type=task_type),
        )
        return [_normalize(np.asarray(embedding.values, dtype=float)) for embedding in response.embeddings]

    def _check_budget(self, missing: int) -> None:
        if missing == 0:
            return
        if self.dry_run:
            raise VertexEmbeddingBudgetError(
                f"dry_run_no_api_call: Vertex embedding cache misses ({missing}) would require live API fetches",
                missing=missing,
                allowed=self.max_new_texts,
                dry_run=True,
            )
        if not self.allow_api:
            raise VertexEmbeddingBudgetError(
                "Vertex embedding cache misses require allow_api=True before live API calls",
                missing=missing,
                allowed=self.max_new_texts,
                dry_run=False,
            )
        remaining = self.max_new_texts - self.actual_new_texts
        if missing > remaining:
            raise VertexEmbeddingBudgetError(
                f"Vertex embedding cache misses ({missing}) exceed max_new_texts remaining budget ({remaining})",
                missing=missing,
                allowed=self.max_new_texts,
                dry_run=False,
            )

    def _live_client(self) -> object:
        if self._client is not None:
            return self._client
        from google import genai

        load_dotenv(self.project_root / ".env")
        credentials = Path(os.environ["GOOGLE_APPLICATION_CREDENTIALS"]).expanduser()
        if not credentials.is_absolute():
            credentials = self.project_root / credentials
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(credentials)
        self._client = genai.Client(
            vertexai=True,
            project=os.environ["GOOGLE_CLOUD_PROJECT"],
            location=os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1"),
        )
        return self._client

    def _load_cache(self) -> None:
        if not self.cache_path.exists():
            return
        for line in self.cache_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = (row["model"], row["task_type"], row["text_hash"])
            self._cache[key] = np.asarray(row["embedding"], dtype=float)

    def _append_cache(self, key: tuple[str, str, str], vector: np.ndarray) -> None:
        model, task_type, text_hash = key
        with self.cache_path.open("a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "model": model,
                        "task_type": task_type,
                        "text_hash": text_hash,
                        "embedding": vector.tolist(),
                    }
                )
                + "\n"
            )

    def _key(self, text: str, task_type: str) -> tuple[str, str, str]:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return (self.model, task_type, digest)


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        return vector
    return vector / norm
