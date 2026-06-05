from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _run_script(script: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / script), *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_repeated_selection_rejects_seed_abbreviation(tmp_path: Path) -> None:
    result = _run_script(
        "run_repeated_selection.py",
        "--dataset",
        "nfcorpus",
        "--data-path",
        str(tmp_path / "missing"),
        "--seed",
        "41",
    )

    assert result.returncode == 2
    assert "unrecognized arguments: --seed" in result.stderr


def test_repeated_gemini_rejects_seed_abbreviation(tmp_path: Path) -> None:
    result = _run_script(
        "run_repeated_gemini_baseline.py",
        "--data-path",
        str(tmp_path / "missing.json"),
        "--seed",
        "41",
    )

    assert result.returncode == 2
    assert "unrecognized arguments: --seed" in result.stderr
