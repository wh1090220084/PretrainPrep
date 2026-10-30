"""Tokenizer adapters and fingerprints used by L1/L2 and manifest binding."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional

from ebook_pipeline.config import Config
from ebook_pipeline.errors import ConfigError
from ebook_pipeline.io_utils import canonical_json, file_sha256_list, sha256_text


class TokenizerAdapter(ABC):
    """Minimal deterministic tokenizer contract consumed by the pipeline."""

    def __init__(self, name: str, eos_ids: List[int]) -> None:
        if not name:
            raise ConfigError("tokenizer name must not be empty")
        if not eos_ids:
            raise ConfigError("tokenizer must expose exactly one or more EOS ids")
        if any(int(value) < 0 for value in eos_ids):
            raise ConfigError("EOS ids must be non-negative")
        self._name = name
        self._eos_ids = [int(value) for value in eos_ids]

    @property
    def name(self) -> str:
        return self._name

    @property
    def eos_ids(self) -> List[int]:
        return list(self._eos_ids)

    @abstractmethod
    def encode(self, text: str) -> List[int]:
        """Encode ordinary text without implicit BOS/EOS tokens."""

    @abstractmethod
    def decode(self, ids: List[int]) -> str:
        """Decode token ids for span/round-trip audits."""

    @abstractmethod
    def fingerprint(self, config: Config) -> str:
        """Return a stable hash of vocabulary and special-token configuration."""


class ByteTokenizer(TokenizerAdapter):
    """Development-only UTF-8 byte tokenizer with id 256 as EOS."""

    def __init__(self, eos_ids: Optional[List[int]] = None) -> None:
        super().__init__("bytes-v1", eos_ids or [256])

    def encode(self, text: str) -> List[int]:
        return list(text.encode("utf-8"))

    def decode(self, ids: List[int]) -> str:
        values = bytes(int(value) for value in ids if int(value) != 256)
        return values.decode("utf-8", errors="replace")

    def fingerprint(self, config: Config) -> str:
        return sha256_text("bytes-v1:eos=256")


class TokenizersJsonAdapter(TokenizerAdapter):
    """Adapter for a local Hugging Face ``tokenizer.json`` asset."""

    def __init__(self, path: Path, name: str, eos_ids: List[int]) -> None:
        super().__init__(name, eos_ids)
        try:
            from tokenizers import Tokenizer
        except ImportError as exc:
            raise ConfigError("tokenizer_json backend requires the 'tokenizers' package") from exc
        self.path = path
        self._tokenizer = Tokenizer.from_file(str(path))

    def encode(self, text: str) -> List[int]:
        return list(self._tokenizer.encode(text, add_special_tokens=False).ids)

    def decode(self, ids: List[int]) -> str:
        return self._tokenizer.decode([int(value) for value in ids], skip_special_tokens=False)

    def fingerprint(self, config: Config) -> str:
        configured_files = [
            config.resolve_path(str(name)) for name in config.get("tokenizer.vocab_files", []) or []
        ]
        files = []
        seen = set()
        for candidate in [self.path] + configured_files:
            resolved = candidate.resolve()
            if resolved not in seen:
                seen.add(resolved)
                files.append(resolved)
        payload = {
            "backend": "tokenizer_json",
            "name": self.name,
            "eos_ids": self.eos_ids,
            "asset_hash": file_sha256_list(files),
            "assets": [path.name for path in files],
        }
        return sha256_text(canonical_json(payload))


class HuggingFaceAdapter(TokenizerAdapter):
    """Adapter for a Hugging Face tokenizer, preferably loaded offline."""

    def __init__(self, config: Config) -> None:
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise ConfigError(
                "huggingface backend requires transformers; install the 'hf' extra"
            ) from exc
        name = str(config.get("tokenizer.name", ""))
        model_name = str(config.get("tokenizer.hf_model", name) or name)
        local_only = bool(config.get("tokenizer.local_files_only", True))
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            local_files_only=local_only,
            trust_remote_code=False,
        )
        configured = config.get("tokenizer.eos_ids", []) or []
        if configured:
            eos_ids = [int(value) for value in configured]
        elif self._tokenizer.eos_token_id is not None:
            eos_ids = [int(self._tokenizer.eos_token_id)]
        else:
            raise ConfigError("tokenizer has no EOS id; set tokenizer.eos_ids explicitly")
        super().__init__(name, eos_ids)

    def encode(self, text: str) -> List[int]:
        return list(self._tokenizer.encode(text, add_special_tokens=False))

    def decode(self, ids: List[int]) -> str:
        return str(self._tokenizer.decode(ids, skip_special_tokens=False))

    def fingerprint(self, config: Config) -> str:
        vocab_files = [config.resolve_path(str(name)) for name in config.get("tokenizer.vocab_files", [])]
        if vocab_files:
            asset_hash = file_sha256_list(vocab_files)
        else:
            backend = getattr(self._tokenizer, "backend_tokenizer", None)
            if backend is not None:
                asset_hash = sha256_text(backend.to_str())
            else:
                vocab = self._tokenizer.get_vocab()
                asset_hash = sha256_text(canonical_json(vocab))
        payload = {
            "backend": "huggingface",
            "name": self.name,
            "model": str(config.get("tokenizer.hf_model", self.name)),
            "eos_ids": self.eos_ids,
            "asset_hash": asset_hash,
        }
        return sha256_text(canonical_json(payload))


def load_tokenizer(config: Config) -> TokenizerAdapter:
    """Construct the exact tokenizer declared by the configuration."""

    backend = str(config.get("tokenizer.backend", "bytes"))
    if backend == "bytes":
        return ByteTokenizer([int(value) for value in config.get("tokenizer.eos_ids", [256])])
    if backend == "tokenizer_json":
        value = config.get("tokenizer.tokenizer_json")
        if not value:
            raise ConfigError("tokenizer.tokenizer_json is required")
        path = config.resolve_path(str(value))
        if not path.exists():
            raise ConfigError(f"tokenizer JSON not found: {path}")
        eos_ids = [int(value) for value in config.get("tokenizer.eos_ids", [])]
        if not eos_ids:
            raise ConfigError("tokenizer.eos_ids is required for tokenizer_json backend")
        return TokenizersJsonAdapter(path, str(config.get("tokenizer.name", path.stem)), eos_ids)
    if backend == "huggingface":
        return HuggingFaceAdapter(config)
    raise ConfigError(f"unsupported tokenizer backend: {backend}")
