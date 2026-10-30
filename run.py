#!/usr/bin/env python3
"""Cross-platform launcher for the TXT-to-JSONL pipeline.

The script locates the repository from its own path, prefers the project
virtual environment when present, adds ``src`` to Python's import path, and
defaults to ``configs/production.yaml run``. Additional CLI arguments are
forwarded to ``ebook_pipeline.cli``.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

KNOWN_COMMANDS = {
    "init",
    "run",
    "stage",
    "validate",
    "lineage",
    "config-hash",
    "summary",
    "repro-check",
    "export-jsonl",
}


def _venv_python(root: Path) -> Optional[Path]:
    candidates = [root / ".venv" / "Scripts" / "python.exe", root / ".venv" / "bin" / "python"]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _maybe_reexec_in_venv(root: Path) -> None:
    """Run through the project venv when the launcher used another Python."""

    if os.environ.get("EBOOK_PIPELINE_VENV_REEXEC") == "1":
        return
    venv_python = _venv_python(root)
    if venv_python is None:
        return
    try:
        same_interpreter = Path(sys.executable).resolve() == venv_python.resolve()
    except OSError:
        same_interpreter = False
    if same_interpreter:
        return
    environment = os.environ.copy()
    environment["EBOOK_PIPELINE_VENV_REEXEC"] = "1"
    result = subprocess.run(
        [str(venv_python), str(Path(__file__).resolve()), *sys.argv[1:]],
        env=environment,
    )
    raise SystemExit(result.returncode)


def _commands_from_argv(arguments: List[str]) -> List[str]:
    """Default bare options such as ``--force`` to the ``run`` command."""

    if not arguments:
        return ["run"]
    if arguments[0] in KNOWN_COMMANDS:
        return arguments
    return ["run", *arguments]


def main() -> int:
    root = Path(__file__).resolve().parent
    _maybe_reexec_in_venv(root)
    source_dir = root / "src"
    if str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

    config = os.environ.get("EBOOK_PIPELINE_CONFIG", str(root / "configs" / "production.yaml"))
    from ebook_pipeline.cli import main as pipeline_main

    return int(pipeline_main(["--config", config, *_commands_from_argv(sys.argv[1:])]))


if __name__ == "__main__":
    raise SystemExit(main())
