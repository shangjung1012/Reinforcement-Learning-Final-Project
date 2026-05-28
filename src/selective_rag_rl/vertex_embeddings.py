from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Protocol

import numpy as np
from dotenv import load_dotenv


class SemanticEmbedder(Protocol):
    def embed_text(self, text: str, task_type: str) -> np.ndarray:
        ...

    def embed_texts(self, texts: list[str], task_type: str) -> list[np.ndarray]:
        ...


class VertexTextEmbeddingProvider:
    def __init__(
        self,
        project_root: Path,
        cache_path: Path,
        model: str = "gemini-embedding-001",
        batch_size: int = 32,
    ) -> None:
        from google import genai

        load_dotenv(project_root / ".env")
        credentials = Path(os.environ["GOOGLE_APPLICATION_CREDENTIALS"]).expanduser()
        if not credentials.is_absolute():
            credentials = project_root / credentials
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(credentials)

        self.model = model
        self.batch_size = batch_size
        self.cache_path = cache_path
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.client = genai.Client(
            vertexai=True,
            project=os.environ["GOOGLE_CLOUD_PROJECT"],
            location=os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1"),
        )
        self._cache: dict[tuple[str, str, str], np.ndarray] = {}
        self._load_cache()

    def embed_text(self, text: str, task_type: str) -> np.ndarray:
        return self.embed_texts([text], task_type=task_type)[0]

    def embed_texts(self, texts: list[str], task_type: str) -> list[np.ndarray]:
        normalized_texts = [text.strip() for text in texts]
        missing = [text for text in dict.fromkeys(normalized_texts) if self._key(text, task_type) not in self._cache]
        for start in range(0, len(missing), self.batch_size):
            batch = missing[start : start + self.batch_size]
            vectors = self._fetch(batch, task_type)
            for text, vector in zip(batch, vectors, strict=True):
                key = self._key(text, task_type)
                self._cache[key] = vector
                self._append_cache(key, vector)
        return [self._cache[self._key(text, task_type)] for text in normalized_texts]

    def _fetch(self, texts: list[str], task_type: str) -> list[np.ndarray]:
        from google.genai import types

        response = self.client.models.embed_content(
            model=self.model,
            contents=texts,
            config=types.EmbedContentConfig(task_type=task_type),
        )
        return [_normalize(np.asarray(embedding.values, dtype=float)) for embedding in response.embeddings]

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
