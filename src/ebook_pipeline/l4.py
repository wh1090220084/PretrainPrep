"""L4 final JSONL shard writer and compact byte-offset index."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from ebook_pipeline.config import Config
from ebook_pipeline.errors import ValidationError
from ebook_pipeline.io_utils import (
    FileLock,
    JSONLWriter,
    atomic_write_json,
    iter_jsonl,
    atomic_write_jsonl,
    read_json,
    read_stage_manifest,
    sha256_file,
    stage_is_done,
    utc_now_iso,
    write_stage_manifest,
    write_stage_marker,
)


def _read_indexed_jsonl(path: Path, offset: int, byte_length: int) -> Dict[str, Any]:
    with path.open("rb") as handle:
        handle.seek(offset)
        line = handle.readline(byte_length)
    if len(line) != byte_length:
        raise ValidationError(f"short read: {path}@{offset}+{byte_length}")
    value = json.loads(line.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValidationError(f"non-object JSONL record: {path}@{offset}")
    return value


def _validate_pack(record: Dict[str, Any], bin_size: int) -> None:
    input_ids = record.get("input_ids", [])
    doc_lengths = record.get("doc_lengths", [])
    position_ids = record.get("position_ids", [])
    loss_mask = record.get("loss_mask", [])
    if len(input_ids) > bin_size:
        raise ValidationError(f"pack overflow: {record.get('pack_id')} {len(input_ids)} > {bin_size}")
    if sum(int(value) for value in doc_lengths) != len(input_ids):
        raise ValidationError(f"doc_lengths mismatch: {record.get('pack_id')}")
    if len(position_ids) != len(input_ids) or len(loss_mask) != len(input_ids):
        raise ValidationError(f"sidecar length mismatch: {record.get('pack_id')}")
    if len(record.get("doc_ids", [])) != len(doc_lengths):
        raise ValidationError(f"doc_ids mismatch: {record.get('pack_id')}")


def run_l4(config: Config, stage_hash: str, config_hash: str, *, force: bool = False) -> Dict[str, Any]:
    """Consolidate packed JSONL into deterministic token-budgeted shards."""

    config.ensure_directories()
    lock_path = config.dirs["state"] / "locks" / "l4.lock"
    with FileLock(lock_path, timeout_seconds=float(config.get("runtime.lock_timeout_seconds", 0))):
        metrics_path = config.dirs["metrics"] / "l4.json"
        if not force and stage_is_done(config, "l4", stage_hash) and metrics_path.exists():
            return read_json(metrics_path)
        l3_manifest = read_stage_manifest(config, "l3")
        packed_index = config.work_dir / str(l3_manifest["index"])
        shard_size = int(config.get("l4.shard_size_tokens", 2_000_000_000))
        bin_size = int(config.get("l3.bin_size", 32700))
        shard_index_path = config.dirs["shards"] / "index.jsonl"
        shard_id = 0
        shard_tokens = 0
        total_tokens = 0
        total_packs = 0
        shard_manifests: List[Dict[str, Any]] = []
        writer: Optional[JSONLWriter] = None
        current_shard_path: Optional[Path] = None

        def open_shard(next_id: int) -> JSONLWriter:
            path = config.dirs["shards"] / f"shard_{next_id:05d}.jsonl"
            instance = JSONLWriter(path)
            instance.__enter__()
            return instance

        try:
            with JSONLWriter(shard_index_path) as index_writer:
                writer = open_shard(shard_id)
                current_shard_path = config.dirs["shards"] / f"shard_{shard_id:05d}.jsonl"
                for packed_ref in iter_jsonl(packed_index):
                    pack_tokens = int(packed_ref["n_tokens"])
                    if shard_tokens > 0 and shard_tokens + pack_tokens > shard_size:
                        writer.close()
                        shard_manifests.append(
                            {
                                "shard_id": shard_id,
                                "path": config.relative_to_work(current_shard_path),
                                "tokens": shard_tokens,
                            }
                        )
                        shard_id += 1
                        shard_tokens = 0
                        writer = open_shard(shard_id)
                        current_shard_path = config.dirs["shards"] / f"shard_{shard_id:05d}.jsonl"

                    source = config.work_dir / str(packed_ref["file"])
                    record = _read_indexed_jsonl(source, int(packed_ref["offset"]), int(packed_ref["byte_length"]))
                    _validate_pack(record, bin_size)
                    if record.get("pack_id") != packed_ref.get("pack_id"):
                        raise ValidationError(f"pack index mismatch: {packed_ref.get('pack_id')}")
                    offset, byte_length = writer.write(record)
                    index_writer.write(
                        {
                            "pack_id": record["pack_id"],
                            "shard_id": shard_id,
                            "file": current_shard_path.name,
                            "offset": offset,
                            "byte_length": byte_length,
                            "n_tokens": len(record["input_ids"]),
                            "n_docs": len(record["doc_ids"]),
                            "cross_doc": bool(record["cross_doc"]),
                            "source_shas": record["source_sha256s"],
                        }
                    )
                    shard_tokens += pack_tokens
                    total_tokens += pack_tokens
                    total_packs += 1
                writer.close()
                shard_manifests.append(
                    {
                        "shard_id": shard_id,
                        "path": config.relative_to_work(current_shard_path),
                        "tokens": shard_tokens,
                    }
                )
        except Exception:
            if writer is not None:
                writer.close(success=False)
            raise

        for shard in shard_manifests:
            path = config.work_dir / shard["path"]
            shard["sha256"] = sha256_file(path)
            shard["bytes"] = path.stat().st_size

        manifest = {
            "shard_count": len(shard_manifests),
            "total_tokens": total_tokens,
            "total_packs": total_packs,
            "bin_size": bin_size,
            "tokenizer": str(config.get("tokenizer.name", "")),
            "config_hash": config_hash,
            "stage_hash": stage_hash,
            "format": "jsonl",
            "shards": shard_manifests,
            "created_at": utc_now_iso(),
        }
        write_stage_manifest(config, "l4", manifest)
        atomic_write_json(config.dirs["shards"] / "manifest.json", manifest)
        metrics = {
            "stage": "l4",
            "format": "jsonl",
            "shard_count": len(shard_manifests),
            "packs": total_packs,
            "tokens": total_tokens,
            "shard_size_tokens": shard_size,
            "shards": shard_manifests,
        }
        atomic_write_json(config.dirs["metrics"] / "l4.json", metrics)
        atomic_write_jsonl(
            config.dirs["audit"] / "l4.jsonl",
            [
                {
                    "stage": "l4",
                    "decision": "accept",
                    "reason": "",
                    "score": 1.0,
                    "config_hash": stage_hash,
                    "shard_count": len(shard_manifests),
                    "total_tokens": total_tokens,
                    "total_packs": total_packs,
                }
            ],
        )
        write_stage_marker(config, "l4", stage_hash)
        return metrics
