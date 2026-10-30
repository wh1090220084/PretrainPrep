"""Release validation, chunk round-trip checks, and pack-to-source lineage."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Dict, List, Set, Tuple

from ebook_pipeline.config import Config
from ebook_pipeline.dataset import PackedDataset
from ebook_pipeline.errors import ValidationError
from ebook_pipeline.io_utils import atomic_write_json, iter_jsonl, read_json, sha256_file
from ebook_pipeline.l3 import ChunkReader
from ebook_pipeline.models import normalized_from_record
from ebook_pipeline.tokenizer import TokenizerAdapter

WHITESPACE_RE = re.compile(r"\s+")
SENTENCE_END = set("。！？!?；;….”’\"')]】》")


def _normalize_whitespace(value: str) -> str:
    return WHITESPACE_RE.sub(" ", value).strip()


def _chunk_refs_for_ids(config: Config, needed_ids: Set[str]) -> Dict[str, Dict[str, Any]]:
    path = config.dirs["chunks"] / "index.jsonl"
    if not path.exists():
        raise ValidationError(f"chunk index missing: {path}")
    refs: Dict[str, Dict[str, Any]] = {}
    for record in iter_jsonl(path):
        chunk_id = str(record["chunk_id"])
        if chunk_id in needed_ids:
            refs[chunk_id] = record
    missing = needed_ids - set(refs)
    if missing:
        raise ValidationError(f"missing chunk lineage ids: {sorted(missing)[:10]}")
    return refs


def _load_shard_index(config: Config) -> List[Dict[str, Any]]:
    path = config.dirs["shards"] / "index.jsonl"
    if not path.exists():
        raise ValidationError(f"shard index missing: {path}")
    return list(iter_jsonl(path))


def validate_pack_record(record: Dict[str, Any], bin_size: int) -> None:
    input_ids = record.get("input_ids")
    doc_lengths = record.get("doc_lengths")
    position_ids = record.get("position_ids")
    loss_mask = record.get("loss_mask")
    doc_ids = record.get("doc_ids")
    sources = record.get("source_sha256s")
    if not isinstance(input_ids, list) or not isinstance(doc_lengths, list):
        raise ValidationError(f"invalid pack arrays: {record.get('pack_id')}")
    if len(input_ids) > bin_size:
        raise ValidationError(f"pack overflow: {record.get('pack_id')}")
    if sum(int(value) for value in doc_lengths) != len(input_ids):
        raise ValidationError(f"doc_lengths mismatch: {record.get('pack_id')}")
    if not isinstance(position_ids, list) or len(position_ids) != len(input_ids):
        raise ValidationError(f"position_ids mismatch: {record.get('pack_id')}")
    if not isinstance(loss_mask, list) or len(loss_mask) != len(input_ids):
        raise ValidationError(f"loss_mask mismatch: {record.get('pack_id')}")
    if not isinstance(doc_ids, list) or len(doc_ids) != len(doc_lengths):
        raise ValidationError(f"doc_ids mismatch: {record.get('pack_id')}")
    if not isinstance(sources, list):
        raise ValidationError(f"source_sha256s mismatch: {record.get('pack_id')}")
    expected_cross = len(set(str(value) for value in sources)) > 1
    if bool(record.get("cross_doc")) != expected_cross:
        raise ValidationError(f"cross_doc mismatch: {record.get('pack_id')}")


def validate_dataset(config: Config, *, sample_packs: int = 10) -> Dict[str, Any]:
    """Validate JSONL shards, index offsets, and sampled source lineage."""

    config.ensure_directories()
    manifest_path = config.dirs["shards"] / "manifest.json"
    if not manifest_path.exists():
        raise ValidationError(f"shard manifest missing: {manifest_path}")
    manifest = read_json(manifest_path)
    bin_size = int(manifest["bin_size"])
    shard_rows = _load_shard_index(config)
    pack_ids: Set[str] = set()
    total_tokens = 0

    for shard in manifest.get("shards", []):
        path = config.work_dir / str(shard["path"])
        if not path.exists():
            raise ValidationError(f"shard missing: {path}")
        digest = sha256_file(path)
        if digest != shard.get("sha256"):
            raise ValidationError(f"shard hash mismatch: {path}")
        if path.stat().st_size != int(shard.get("bytes", -1)):
            raise ValidationError(f"shard size mismatch: {path}")

    for row in shard_rows:
        pack_id = str(row["pack_id"])
        if pack_id in pack_ids:
            raise ValidationError(f"duplicate pack_id: {pack_id}")
        pack_ids.add(pack_id)
        path = config.dirs["shards"] / str(row["file"])
        with path.open("rb") as handle:
            handle.seek(int(row["offset"]))
            payload = handle.read(int(row["byte_length"]))
        if len(payload) != int(row["byte_length"]):
            raise ValidationError(f"short index read for {pack_id}")
        record = json.loads(payload.decode("utf-8"))
        if record.get("pack_id") != pack_id:
            raise ValidationError(f"pack index mismatch: {pack_id}")
        validate_pack_record(record, bin_size)
        if int(row["n_tokens"]) != len(record["input_ids"]):
            raise ValidationError(f"index token count mismatch: {pack_id}")
        total_tokens += int(row["n_tokens"])

    if len(pack_ids) != int(manifest.get("total_packs", -1)):
        raise ValidationError("manifest pack count mismatch")
    if total_tokens != int(manifest.get("total_tokens", -1)):
        raise ValidationError("manifest token count mismatch")

    selected = sorted(shard_rows, key=lambda row: hashlib.sha256(str(row["pack_id"]).encode()).hexdigest())[:sample_packs]
    with PackedDataset(config.dirs["shards"]) as dataset:
        selected_packs = [dataset[str(row["pack_id"])] for row in selected]
    needed_ids: Set[str] = {str(doc_id) for pack in selected_packs for doc_id in pack["doc_ids"]}
    chunk_index = _chunk_refs_for_ids(config, needed_ids)
    lineage_checks: List[Dict[str, Any]] = []
    with ChunkReader(config) as reader:
        for pack in selected_packs:
            for doc_id in pack["doc_ids"]:
                ref = chunk_index[str(doc_id)]
                chunk = reader.read(ref)
                if chunk["source_sha256"] not in pack["source_sha256s"]:
                    raise ValidationError(f"source mismatch for {doc_id}")
                if int(chunk["char_end"]) <= int(chunk["char_start"]):
                    raise ValidationError(f"invalid span for {doc_id}")
                lineage_checks.append(
                    {
                        "pack_id": pack["pack_id"],
                        "doc_id": doc_id,
                        "source_sha256": chunk["source_sha256"],
                        "chapter_path": chunk["chapter_path"],
                        "char_start": chunk["char_start"],
                        "char_end": chunk["char_end"],
                    }
                )

    report = {
        "passed": True,
        "shard_count": len(manifest.get("shards", [])),
        "pack_count": len(pack_ids),
        "total_tokens": total_tokens,
        "lineage_checks": len(lineage_checks),
        "sampled_packs": len(selected),
        "failures": [],
    }
    atomic_write_json(config.dirs["metrics"] / "validation.json", report)
    return report


def _sample_indices(total: int, wanted: int) -> Set[int]:
    if total <= wanted:
        return set(range(total))
    return {int(index * (total / float(wanted))) for index in range(wanted)}


def validate_chunks(
    config: Config,
    tokenizer: TokenizerAdapter,
    *,
    sample_size: int = 1000,
) -> Dict[str, Any]:
    """Check token/span round trips and sentence-boundary quality on a sample."""

    index_path = config.dirs["chunks"] / "index.jsonl"
    if not index_path.exists():
        raise ValidationError(f"chunk index missing: {index_path}")
    total = sum(1 for _ in iter_jsonl(index_path))
    selected = _sample_indices(total, sample_size)
    failures: List[Dict[str, Any]] = []
    boundary_inside_sentence = 0
    checked = 0
    gram_owners: Dict[Tuple[int, ...], str] = {}
    gram_total = 0
    cross_doc_gram_hits = 0

    with ChunkReader(config) as reader:
        for index, ref in enumerate(iter_jsonl(index_path)):
            if index not in selected:
                continue
            chunk = reader.read(ref)
            sha = str(chunk["source_sha256"])
            accepted_path = config.dirs["accepted"] / f"{sha}.json"
            if not accepted_path.exists():
                raise ValidationError(f"accepted record missing for {sha}")
            book = normalized_from_record(read_json(accepted_path), config.work_dir)
            start = int(chunk["char_start"])
            end = int(chunk["char_end"])
            expected = book.text[start:end]
            decoded = tokenizer.decode([int(value) for value in chunk["ids"]])
            same = _normalize_whitespace(decoded) == _normalize_whitespace(expected)
            if not same:
                failures.append(
                    {
                        "chunk_id": chunk["chunk_id"],
                        "reason": "roundtrip_mismatch",
                        "char_start": start,
                        "char_end": end,
                    }
                )
            last_content = expected.rstrip()
            next_char = book.text[end : end + 1]
            if end < len(book.text) and last_content and last_content[-1] not in SENTENCE_END and not next_char.isspace():
                boundary_inside_sentence += 1
            checked += 1
            ids = [int(value) for value in chunk["ids"]]
            if len(ids) >= 8:
                gram_count = len(ids) - 7
                stride = max(1, gram_count // 1000)
                for position in range(0, gram_count, stride):
                    key = tuple(ids[position : position + 8])
                    owner = gram_owners.get(key)
                    if owner is None:
                        gram_owners[key] = sha
                    elif owner != sha:
                        cross_doc_gram_hits += 1
                    gram_total += 1

    cross_doc_gram_ratio = cross_doc_gram_hits / max(1, gram_total)
    boundary_ratio = boundary_inside_sentence / max(1, checked)
    report = {
        "checked": checked,
        "roundtrip_failures": len(failures),
        "boundary_inside_sentence_ratio": boundary_ratio,
        "failures": failures[:100],
        "cross_doc_8gram_repeat_ratio": cross_doc_gram_ratio,
        "cross_doc_8gram_below_1pct": cross_doc_gram_ratio < 0.01,
    }
    atomic_write_json(config.dirs["metrics"] / "chunk_validation.json", report)
    if failures:
        raise ValidationError(f"{len(failures)} chunk round-trip failures")
    return report


def inspect_lineage(config: Config, pack_id: str) -> Dict[str, Any]:
    """Resolve one pack to chunks, normalized text, source path, and source hash."""

    with PackedDataset(config.dirs["shards"]) as dataset:
        pack = dataset[pack_id]
    chunk_index = _chunk_refs_for_ids(config, {str(doc_id) for doc_id in pack["doc_ids"]})
    details: List[Dict[str, Any]] = []
    with ChunkReader(config) as reader:
        for doc_id in pack["doc_ids"]:
            if doc_id not in chunk_index:
                raise ValidationError(f"chunk missing from index: {doc_id}")
            ref = chunk_index[doc_id]
            chunk = reader.read(ref)
            sha = str(chunk["source_sha256"])
            accepted_path = config.dirs["accepted"] / f"{sha}.json"
            book = normalized_from_record(read_json(accepted_path), config.work_dir)
            source_path = config.books_dir / book.source.path
            computed = sha256_file(source_path) if source_path.exists() else None
            details.append(
                {
                    "chunk_id": doc_id,
                    "chapter_path": chunk["chapter_path"],
                    "char_start": chunk["char_start"],
                    "char_end": chunk["char_end"],
                    "source_sha256": sha,
                    "source_path": str(source_path),
                    "source_exists": source_path.exists(),
                    "source_sha256_matches": computed == sha,
                    "text": book.text[int(chunk["char_start"]) : int(chunk["char_end"])][:500],
                }
            )
    return {
        "pack_id": pack_id,
        "n_tokens": len(pack["input_ids"]),
        "doc_ids": pack["doc_ids"],
        "source_sha256s": pack["source_sha256s"],
        "documents": details,
    }
