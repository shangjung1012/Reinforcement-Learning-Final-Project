from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    load_dotenv(project_root / ".env")

    project = os.environ.get("GOOGLE_CLOUD_PROJECT")
    location = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
    model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    credentials = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")

    if not project:
        print("Missing GOOGLE_CLOUD_PROJECT.", file=sys.stderr)
        return 2
    if not credentials:
        print("Missing GOOGLE_APPLICATION_CREDENTIALS.", file=sys.stderr)
        return 2
    credentials_path = Path(credentials).expanduser()
    if not credentials_path.is_absolute():
        credentials_path = project_root / credentials_path
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(credentials_path)

    client = genai.Client(vertexai=True, project=project, location=location)
    response = client.models.generate_content(
        model=model,
        contents=(
            "Rewrite this question into one concise Wikipedia search query. "
            "Return only the rewritten query.\n\n"
            "Question: Which author lived longer, Nelson Algren or Nathanael West?"
        ),
        config=types.GenerateContentConfig(temperature=0.1),
    )
    print(response.text.strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
