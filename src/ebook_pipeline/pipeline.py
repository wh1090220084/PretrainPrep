"""Stage orchestration with dependency-aware hashes and deterministic reruns."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, Optional

from ebook_pipeline import __version__
from ebook_pipeline.config import Config
from ebook_pipeline.io_utils import canonical_json, read_stage_manifest, stable_hash
from ebook_pipeline.l0 import run_l0
from ebook_pipeline.l1 import run_l1
from ebook_pipeline.l2 import run_l2
from ebook_pipeline.l3 import run_l3
from ebook_pipeline.l4 import run_l4
from ebook_pipeline.tokenizer import TokenizerAdapter, load_tokenizer
from ebook_pipeline.validation import validate_chunks, validate_dataset



def _code_fingerprint() -> str:
    """Hash all package Python sources so code changes invalidate stage markers."""

    digest = hashlib.sha256()
    package_dir = Path(__file__).resolve().parent
    for path in sorted(package_dir.glob("*.py")):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


CODE_FINGERPRINT = _code_fingerprint()
STAGES = ["l0", "l1", "l2", "l3", "l4"]


def _manifest_digest(manifest: Dict[str, Any]) -> str:
    payload = dict(manifest)
    payload.pop("stage_hash", None)
    return stable_hash(payload)


def dependent_stage_hash(
    config: Config,
    stage: str,
    tokenizer_fingerprint: str,
    upstream_manifest: Optional[Dict[str, Any]],
) -> str:
    """Bind a stage to config, code version, tokenizer, and upstream artifacts."""

    payload = {
        "base": config.stage_hash(stage, tokenizer_fingerprint),
        "pipeline_version": __version__,
        "pipeline_code_fingerprint": CODE_FINGERPRINT,
        "upstream_manifest": _manifest_digest(upstream_manifest) if upstream_manifest else None,
    }
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


class Pipeline:
    """High-level runner that keeps tokenizer and hashes consistent across layers."""

    def __init__(self, config: Config, *, force: bool = False) -> None:
        self.config = config
        self.force = force
        self.tokenizer: TokenizerAdapter = load_tokenizer(config)
        self.tokenizer_fingerprint = self.tokenizer.fingerprint(config)
        self.config_hash = config.config_hash(self.tokenizer_fingerprint)

    def stage(self, stage: str) -> Dict[str, Any]:
        if stage not in STAGES:
            raise ValueError(f"unknown stage: {stage}")
        self.config.ensure_directories()
        if stage == "l0":
            stage_hash = dependent_stage_hash(self.config, "l0", self.tokenizer_fingerprint, None)
            return run_l0(self.config, stage_hash, force=self.force)

        previous = STAGES[STAGES.index(stage) - 1]
        upstream = read_stage_manifest(self.config, previous)
        stage_hash = dependent_stage_hash(self.config, stage, self.tokenizer_fingerprint, upstream)
        if stage == "l1":
            return run_l1(self.config, self.tokenizer, stage_hash, force=self.force)
        if stage == "l2":
            return run_l2(self.config, self.tokenizer, stage_hash, force=self.force)
        if stage == "l3":
            return run_l3(self.config, self.tokenizer, stage_hash, force=self.force)
        if stage == "l4":
            return run_l4(self.config, stage_hash, self.config_hash, force=self.force)
        raise AssertionError(stage)

    def run_all(self, *, validate: Optional[bool] = None) -> Dict[str, Any]:
        """Run L0-L4, then optional release-level validation and metric summary."""

        from ebook_pipeline.summary import summarize

        results: Dict[str, Any] = {}
        for stage in STAGES:
            results[stage] = self.stage(stage)
        should_validate = (
            bool(self.config.get("runtime.validate_after_run", True)) if validate is None else bool(validate)
        )
        if should_validate:
            sample_packs = int(self.config.get("runtime.validation_sample_packs", 10))
            sample_chunks = int(self.config.get("runtime.validation_sample_chunks", 1000))
            results["validation"] = validate_dataset(self.config, sample_packs=sample_packs)
            results["chunk_validation"] = validate_chunks(
                self.config,
                self.tokenizer,
                sample_size=sample_chunks,
            )
        results["summary"] = summarize(self.config, self.config_hash)
        return results

    def run_stage(self, stage: str, *, validate: bool = False) -> Dict[str, Any]:
        result = self.stage(stage)
        if validate and stage == "l4":
            result = {
                "l4": result,
                "validation": validate_dataset(self.config),
                "chunk_validation": validate_chunks(self.config, self.tokenizer, sample_size=1000),
            }
        return result
