"""Full-pipeline determinism check using an isolated alternate work directory."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict

import yaml

from ebook_pipeline.config import Config, load_config
from ebook_pipeline.io_utils import atomic_write_json, sha256_file
from ebook_pipeline.pipeline import Pipeline


def _artifact_hashes(dataset_dir: Path) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for path in sorted(dataset_dir.glob("shard_*.jsonl")):
        result[path.name] = sha256_file(path)
    index = dataset_dir / "index.jsonl"
    if index.exists():
        result["index.jsonl"] = sha256_file(index)
    return result


def reproduce_and_compare(config: Config, alternate_work_dir: Path) -> Dict[str, Any]:
    """Run Pipeline in an isolated work dir and compare final JSONL bytes."""

    primary = _artifact_hashes(config.dirs["shards"])
    if not primary:
        raise FileNotFoundError("primary shards are missing; run the pipeline first")
    alternate_work_dir = alternate_work_dir.resolve()
    raw = yaml.safe_load(config.source_path.read_text(encoding="utf-8")) or {}
    raw.setdefault("paths", {})["work_dir"] = str(alternate_work_dir)
    temporary_path: Path
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".yaml",
        prefix=".repro-",
        dir=str(config.base_dir),
        delete=False,
    ) as handle:
        yaml.safe_dump(raw, handle, allow_unicode=True, sort_keys=False)
        temporary_path = Path(handle.name)
    try:
        alternate_config = load_config(temporary_path)
        Pipeline(alternate_config, force=True).run_all(validate=False)
        alternate = _artifact_hashes(alternate_config.dirs["shards"])
    finally:
        temporary_path.unlink(missing_ok=True)
    differing = sorted(
        name for name in set(primary) | set(alternate) if primary.get(name) != alternate.get(name)
    )
    result = {
        "passed": not differing,
        "primary_artifacts": primary,
        "alternate_artifacts": alternate,
        "alternate_work_dir": str(alternate_work_dir),
        "differing_artifacts": differing,
    }
    atomic_write_json(config.dirs["metrics"] / "determinism.json", result)
    return result
