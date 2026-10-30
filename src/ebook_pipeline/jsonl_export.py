"""Export cleaned pretraining text as generic ``{"text": ...}`` JSONL."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterator

from ebook_pipeline.config import Config
from ebook_pipeline.errors import StateError
from ebook_pipeline.io_utils import atomic_write_json, atomic_write_jsonl, read_json, read_stage_manifest
from ebook_pipeline.l2 import _content_segments, split_span
from ebook_pipeline.models import normalized_from_record
from ebook_pipeline.tokenizer import TokenizerAdapter, TokenizersJsonAdapter


def load_training_tokenizer(path: Path) -> TokenizerAdapter:
    """Load a local training tokenizer directory or tokenizer.json file."""

    path = path.resolve()
    tokenizer_json = path / "tokenizer.json" if path.is_dir() else path
    tokenizer_config_path = tokenizer_json.parent / "tokenizer_config.json"
    if not tokenizer_json.exists():
        raise StateError(f"tokenizer.json not found: {tokenizer_json}")
    if not tokenizer_config_path.exists():
        raise StateError(f"tokenizer_config.json not found: {tokenizer_config_path}")
    config = json.loads(tokenizer_config_path.read_text(encoding="utf-8"))
    eos_value = config.get("eos_token")
    if isinstance(eos_value, dict):
        eos_value = eos_value.get("content")
    try:
        from tokenizers import Tokenizer
    except ImportError as exc:
        raise StateError("JSONL export requires the 'tokenizers' package") from exc
    raw = Tokenizer.from_file(str(tokenizer_json))
    eos_id = raw.token_to_id(str(eos_value)) if eos_value else None
    if eos_id is None:
        raise StateError(f"could not resolve EOS token id from {tokenizer_config_path}")
    return TokenizersJsonAdapter(tokenizer_json, f"local:{tokenizer_json.parent.name}", [int(eos_id)])


def export_text_jsonl(
    config: Config,
    tokenizer: TokenizerAdapter,
    output_path: Path,
    *,
    max_tokens: int,
) -> Dict[str, Any]:
    """Write cleaned text records that are token-budgeted for a training side.

    Each output line contains only ``{"text": ...}``. The caller must pass the
    same tokenizer used by the training framework so the per-sample token budget
    is exact.
    """

    if max_tokens <= 0:
        raise StateError("JSONL export max_tokens must be positive")
    try:
        manifest = read_stage_manifest(config, "l1")
    except StateError as exc:
        raise StateError(
            f"{exc}. Run the full pipeline first, for example: python run.py --force"
        ) from exc
    books = sorted(manifest.get("books", []), key=lambda item: str(item["sha256"]))
    if not books:
        raise StateError("no accepted books available for JSONL export")

    separators = [str(value) for value in config.get("l2.separators", ["\n\n", "\n", "。！？；…", ".!?;", "，,"])]
    stats = {"samples": 0, "tokens": 0, "chars": 0}

    def records() -> Iterator[Dict[str, str]]:
        for item in books:
            accepted_path = config.work_dir / str(item["path"])
            book = normalized_from_record(read_json(accepted_path), config.work_dir)
            for start, end, _chapter_path, _chapter_level in _content_segments(book.structure.root, []):
                start = max(0, min(len(book.text), start))
                end = max(start, min(len(book.text), end))
                if not book.text[start:end].strip():
                    continue
                for ids, part_start, part_end in split_span(
                    book.text,
                    start,
                    end,
                    tokenizer,
                    max_tokens,
                    separators,
                ):
                    text = book.text[part_start:part_end]
                    if not text.strip():
                        continue
                    stats["samples"] += 1
                    stats["tokens"] += len(ids)
                    stats["chars"] += len(text)
                    yield {"text": text}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_jsonl(output_path, records())
    result = {
        "format": "text_jsonl",
        "output": str(output_path),
        "max_tokens_per_sample": max_tokens,
        "tokenizer": tokenizer.name,
        "eos_ids": tokenizer.eos_ids,
        **stats,
    }
    atomic_write_json(config.dirs["metrics"] / "jsonl_export.json", result)
    return result