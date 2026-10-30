"""Configuration loading, validation, path resolution, and hashing."""

from __future__ import annotations

import copy
import importlib.util
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Union

import yaml

from ebook_pipeline.errors import ConfigError
from ebook_pipeline.models import get_nested

SUPPORTED_PROFILES = {"development", "production"}
SUPPORTED_TOKENIZERS = {"bytes", "tokenizer_json", "huggingface"}
STAGE_DIR_NAMES = {
    "normalized": "normalized",
    "quarantine": "quarantine",
    "accepted": "accepted",
    "dedup": "dedup",
    "chunks": "chunks",
    "packed": "packed",
    "shards": "shards",
    "state": "state",
    "audit": "audit",
    "metrics": "metrics",
    "errors": "errors",
}


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class Config:
    """Validated, path-aware view over one YAML configuration file."""

    data: Dict[str, Any]
    source_path: Path

    @property
    def base_dir(self) -> Path:
        return self.source_path.parent

    @property
    def profile(self) -> str:
        return str(self.get("profile", "development"))

    @property
    def work_dir(self) -> Path:
        return self.resolve_path(str(self.get("paths.work_dir", "data")))

    @property
    def books_dir(self) -> Path:
        return self.resolve_path(str(self.get("paths.books_dir", "books")))

    def get(self, dotted_key: str, default: Any = None) -> Any:
        return get_nested(self.data, dotted_key, default)

    def section(self, name: str) -> Dict[str, Any]:
        value = self.data.get(name, {})
        if not isinstance(value, dict):
            raise ConfigError(f"config section '{name}' must be a mapping")
        return copy.deepcopy(value)

    def resolve_path(self, value: str, *, relative_to: Optional[Path] = None) -> Path:
        """Resolve a configured path relative to the config file or a given root."""

        path = Path(os.path.expandvars(os.path.expanduser(value)))
        if path.is_absolute():
            return path
        return (relative_to or self.base_dir) / path

    @property
    def dirs(self) -> Dict[str, Path]:
        return {name: self.work_dir / dirname for name, dirname in STAGE_DIR_NAMES.items()}

    def ensure_directories(self) -> None:
        self.work_dir.mkdir(parents=True, exist_ok=True)
        for path in self.dirs.values():
            path.mkdir(parents=True, exist_ok=True)
        (self.dirs["audit"] / "records").mkdir(parents=True, exist_ok=True)
        (self.dirs["state"] / "locks").mkdir(parents=True, exist_ok=True)

    def relative_to_work(self, path: Path) -> str:
        """Return a POSIX-style relative path when possible, else an absolute path."""

        try:
            return path.resolve().relative_to(self.work_dir.resolve()).as_posix()
        except ValueError:
            return str(path.resolve())

    def config_hash(self, tokenizer_fingerprint: str) -> str:
        """Hash the canonical config together with the tokenizer asset fingerprint."""

        payload = {
            "schema": 1,
            "config": self.data,
            "tokenizer_fingerprint": tokenizer_fingerprint,
        }
        return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()

    def stage_hash(self, stage: str, tokenizer_fingerprint: str) -> str:
        """Hash a stage-relevant subset to avoid needless invalidation across layers."""

        shared = {
            "profile": self.get("profile"),
            "tokenizer": self.get("tokenizer", {}),
            "determinism": self.get("determinism", {}),
        }
        payload = {
            "schema": 1,
            "stage": stage,
            "shared": shared,
            "stage_config": self.get(stage, {}),
            "tokenizer_fingerprint": tokenizer_fingerprint,
        }
        return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()

    def validate(self, *, require_runtime_assets: bool = True) -> None:
        """Validate structural constraints and production safety requirements."""

        if self.profile not in SUPPORTED_PROFILES:
            raise ConfigError(f"unsupported profile: {self.profile}")

        backend = str(self.get("tokenizer.backend", "bytes"))
        if backend not in SUPPORTED_TOKENIZERS:
            raise ConfigError(f"unsupported tokenizer backend: {backend}")

        bin_size = int(self.get("l3.bin_size", 0))
        max_tokens = int(self.get("l2.max_tokens", 0))
        min_tokens = int(self.get("l2.min_tokens", 0))
        shard_tokens = int(self.get("l4.shard_size_tokens", 0))
        if bin_size <= 0:
            raise ConfigError("l3.bin_size must be positive")
        if max_tokens <= 0 or min_tokens <= 0:
            raise ConfigError("l2.max_tokens and l2.min_tokens must be positive")
        if min_tokens > max_tokens:
            raise ConfigError("l2.min_tokens must not exceed l2.max_tokens")
        if max_tokens > bin_size:
            raise ConfigError("l2.max_tokens must not exceed l3.bin_size")
        if shard_tokens < bin_size:
            raise ConfigError("l4.shard_size_tokens must be >= l3.bin_size")

        if str(self.get("l3.algorithm", "obfd")) != "obfd":
            raise ConfigError("only the OBFD packing algorithm is supported")
        if str(self.get("l3.pack_mode", "bfd")) not in {"bfd", "per_book"}:
            raise ConfigError("l3.pack_mode must be bfd or per_book")
        if int(self.get("l3.max_chunks_in_memory", 0)) <= 0:
            raise ConfigError("l3.max_chunks_in_memory must be positive")
        if int(self.get("l3.external_sort_runs", 0)) <= 0:
            raise ConfigError("l3.external_sort_runs must be positive")
        if str(self.get("l4.format", "jsonl")).lower() != "jsonl":
            raise ConfigError("l4.format must be jsonl for this implementation")
        configured_eos = self.get("tokenizer.eos_ids", []) or []
        if configured_eos and max_tokens + len(configured_eos) > bin_size:
            raise ConfigError("l2.max_tokens + len(tokenizer.eos_ids) must not exceed l3.bin_size")
        min_chars = int(self.get("l0.min_chars", 0))
        if min_chars < 0:
            raise ConfigError("l0.min_chars must be non-negative")
        hamming = int(self.get("l1.simhash_hamming", 0))
        bits = int(self.get("l1.simhash_bits", 0))
        if bits <= 0 or bits % 8 != 0:
            raise ConfigError("l1.simhash_bits must be a positive multiple of 8")
        if hamming < 0 or hamming > bits:
            raise ConfigError("l1.simhash_hamming is out of range")
        bands = int(self.get("l1.simhash_bands", 8))
        if bands <= 0 or bits % bands != 0:
            raise ConfigError("l1.simhash_bits must be divisible by simhash_bands")

        weights = self.get("l1.stat_weights", {})
        if not isinstance(weights, dict):
            raise ConfigError("l1.stat_weights must be a mapping")

        if self.profile == "production":
            if backend not in {"huggingface", "tokenizer_json"}:
                raise ConfigError(
                    "production profile requires a real tokenizer backend "
                    "(huggingface or tokenizer_json), not bytes"
                )
            if not str(self.get("tokenizer.name", "")).strip():
                raise ConfigError("production profile requires tokenizer.name")
            vocab_files: Iterable[str] = self.get("tokenizer.vocab_files", []) or []
            tokenizer_json = self.get("tokenizer.tokenizer_json")
            if not list(vocab_files) and not tokenizer_json:
                raise ConfigError(
                    "production profile requires tokenizer.vocab_files or tokenizer.tokenizer_json "
                    "so config_hash binds the exact vocabulary"
                )
            detector = str(self.get("l1.language.detector", "heuristic"))
            if detector not in {"fasttext", "auto", "heuristic"}:
                raise ConfigError(f"unsupported language detector: {detector}")
            require_fasttext = bool(self.get("l1.language.require_fasttext", False))
            strict_fasttext = require_fasttext or (
                detector == "fasttext" and not bool(self.get("l1.language.heuristic_fallback", False))
            )
            if require_runtime_assets and strict_fasttext:
                if detector == "heuristic":
                    raise ConfigError("require_fasttext is true but detector is heuristic")
                if importlib.util.find_spec("fasttext") is None:
                    raise ConfigError("require_fasttext is true but the fasttext package is not installed")
                model_path = self.get("l1.language.model_path")
                if not model_path:
                    raise ConfigError("fastText detector requires l1.language.model_path")
                resolved = self.resolve_path(str(model_path))
                if not resolved.exists():
                    raise ConfigError(f"fastText model not found: {resolved}")

    def validate_assets(self) -> None:
        """Check required runtime files without enforcing production-only rules."""

        if not self.books_dir.exists():
            raise ConfigError(f"books directory does not exist: {self.books_dir}")
        backend = str(self.get("tokenizer.backend", "bytes"))
        if backend == "tokenizer_json":
            value = self.get("tokenizer.tokenizer_json")
            if not value:
                raise ConfigError("tokenizer.tokenizer_json is required")
            if not self.resolve_path(str(value)).exists():
                raise ConfigError(f"tokenizer JSON not found: {self.resolve_path(str(value))}")
        vocab_files = self.get("tokenizer.vocab_files", []) or []
        for name in vocab_files:
            path = self.resolve_path(str(name))
            if not path.exists():
                raise ConfigError(f"tokenizer vocab file not found: {path}")


def load_config(path: Union[str, Path], *, profile: Optional[str] = None) -> Config:
    """Load YAML and optionally override its profile before validation."""

    config_path = Path(path).expanduser().resolve()
    if not config_path.exists():
        raise ConfigError(f"config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ConfigError("config root must be a mapping")
    if profile is not None:
        raw["profile"] = profile
    config = Config(data=raw, source_path=config_path)
    config.validate(require_runtime_assets=False)
    return config
