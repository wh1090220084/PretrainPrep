"""L2 ids-space semantic chunking with exact character spans and short-chunk merge."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from ebook_pipeline.config import Config
from ebook_pipeline.errors import PackingError
from ebook_pipeline.io_utils import (
    FileLock,
    JSONLWriter,
    atomic_write_json,
    atomic_write_jsonl,
    compile_audit,
    read_json,
    read_jsonl,
    read_stage_manifest,
    stage_is_done,
    write_audit_record,
    write_stage_manifest,
    write_stage_marker,
)
from ebook_pipeline.models import NormalizedBook, StructureNode, TokenizedChunk, normalized_from_record
from ebook_pipeline.tokenizer import TokenizerAdapter

SpanPart = Tuple[List[int], int, int]


def _separator_positions(segment: str, separator: str) -> List[int]:
    """Return absolute-in-segment split positions after each separator."""

    if not separator:
        return []
    if separator == "\n\n":
        return [match.end() for match in re.finditer(r"\n{2,}", segment)]
    if separator in {"。！？；…", ".!?;", "，,"}:
        return [match.end() for match in re.finditer(f"[{re.escape(separator)}]", segment)]
    positions: List[int] = []
    start = 0
    while True:
        index = segment.find(separator, start)
        if index < 0:
            break
        positions.append(index + len(separator))
        start = index + len(separator)
    return positions


def _hard_split(
    text: str,
    start: int,
    end: int,
    tokenizer: TokenizerAdapter,
    max_tokens: int,
    separators: Sequence[str],
) -> List[SpanPart]:
    """Binary-search a character boundary that fits the token budget."""

    low = start + 1
    high = end
    best = start + 1
    while low <= high:
        middle = (low + high) // 2
        count = len(tokenizer.encode(text[start:middle]))
        if count <= max_tokens:
            best = middle
            low = middle + 1
        else:
            high = middle - 1
    left_ids = tokenizer.encode(text[start:best])
    while len(left_ids) > max_tokens and best > start + 1:
        best -= 1
        left_ids = tokenizer.encode(text[start:best])
    if len(left_ids) > max_tokens:
        raise PackingError(
            f"single character exceeds max_tokens at {start}: {len(left_ids)} > {max_tokens}"
        )
    parts: List[SpanPart] = [(left_ids, start, best)]
    if best < end:
        parts.extend(_partition_range(text, best, end, tokenizer, max_tokens, separators, 0))
    return parts


def _partition_range(
    text: str,
    start: int,
    end: int,
    tokenizer: TokenizerAdapter,
    max_tokens: int,
    separators: Sequence[str],
    separator_index: int,
) -> List[SpanPart]:
    """Recursively split an oversized span by ordered semantic separators."""

    if start >= end:
        return []
    ids = tokenizer.encode(text[start:end])
    if len(ids) <= max_tokens:
        return [(ids, start, end)]
    if separator_index >= len(separators):
        return _hard_split(text, start, end, tokenizer, max_tokens, separators)

    separator = separators[separator_index]
    segment = text[start:end]
    positions = [start + position for position in _separator_positions(segment, separator)]
    positions = sorted({position for position in positions if start < position < end})
    if not positions:
        return _partition_range(text, start, end, tokenizer, max_tokens, separators, separator_index + 1)

    pieces: List[SpanPart] = []
    cursor = start
    for position in positions + [end]:
        if position <= cursor:
            continue
        pieces.extend(
            _partition_range(
                text,
                cursor,
                position,
                tokenizer,
                max_tokens,
                separators,
                separator_index + 1,
            )
        )
        cursor = position
    return pieces


def split_span(
    text: str,
    start: int,
    end: int,
    tokenizer: TokenizerAdapter,
    max_tokens: int,
    separators: Sequence[str],
) -> List[SpanPart]:
    """Split one structure leaf into token-bounded parts.

    The returned spans are contiguous and non-overlapping. Adjacent parts are
    greedily merged back up to ``max_tokens`` so a paragraph does not become a
    chunk solely because it crossed an earlier recursive boundary.
    """

    leaves = _partition_range(text, start, end, tokenizer, max_tokens, separators, 0)
    merged: List[SpanPart] = []
    for ids, part_start, part_end in leaves:
        if merged and len(merged[-1][0]) + len(ids) <= max_tokens:
            previous_ids, previous_start, _previous_end = merged[-1]
            merged[-1] = (previous_ids + ids, previous_start, part_end)
        else:
            merged.append((ids, part_start, part_end))
    return merged


def _merge_chunks(left: TokenizedChunk, right: TokenizedChunk, *, cross_chapter: bool) -> TokenizedChunk:
    flags = sorted(set(left.flags + right.flags + ["short_chunk_merged"]))
    if cross_chapter:
        flags.append("cross_chapter_merge")
    ids = left.ids + right.ids
    return TokenizedChunk(
        chunk_id=left.chunk_id,
        seq=left.seq,
        source_sha256=left.source_sha256,
        book_title=left.book_title,
        chapter_path=list(left.chapter_path),
        chapter_level=left.chapter_level if not cross_chapter else min(left.chapter_level, right.chapter_level),
        char_start=left.char_start,
        char_end=right.char_end,
        ids=ids,
        n_tokens=len(ids),
        flags=sorted(set(flags)),
    )


def merge_short_chunks(
    chunks: Sequence[TokenizedChunk],
    *,
    min_tokens: int,
    max_tokens: int,
    allow_cross_chapter: bool = True,
) -> List[TokenizedChunk]:
    """Merge undersized neighbors without ever silently discarding a chunk."""

    same_chapter: List[TokenizedChunk] = []
    for chunk in chunks:
        if (
            same_chapter
            and same_chapter[-1].chapter_path == chunk.chapter_path
            and (same_chapter[-1].n_tokens < min_tokens or chunk.n_tokens < min_tokens)
            and same_chapter[-1].n_tokens + chunk.n_tokens <= max_tokens
        ):
            same_chapter[-1] = _merge_chunks(same_chapter[-1], chunk, cross_chapter=False)
        else:
            same_chapter.append(chunk)

    if not allow_cross_chapter:
        return same_chapter

    cross_merged: List[TokenizedChunk] = []
    for chunk in same_chapter:
        if (
            cross_merged
            and cross_merged[-1].chapter_path != chunk.chapter_path
            and (cross_merged[-1].n_tokens < min_tokens or chunk.n_tokens < min_tokens)
            and cross_merged[-1].n_tokens + chunk.n_tokens <= max_tokens
        ):
            cross_merged[-1] = _merge_chunks(cross_merged[-1], chunk, cross_chapter=True)
        else:
            cross_merged.append(chunk)
    return cross_merged


def _content_segments(
    node: StructureNode,
    path: List[str],
) -> List[Tuple[int, int, List[str], int]]:
    """Yield coverage for parent preambles, child gaps, and leaf content."""

    node_path = path + ([node.title] if node.level >= 0 else [])
    start = max(0, int(node.char_start))
    end = max(start, int(node.char_end))
    if not node.children:
        return [(start, end, node_path or ["正文"], max(1, node.level))]

    segments: List[Tuple[int, int, List[str], int]] = []
    cursor = start
    for child in sorted(node.children, key=lambda value: value.char_start):
        child_start = max(start, min(end, int(child.char_start)))
        if cursor < child_start:
            segments.append((cursor, child_start, node_path or ["正文"], max(1, node.level)))
        segments.extend(_content_segments(child, node_path))
        cursor = max(cursor, min(end, int(child.char_end)), child_start)
    if cursor < end:
        segments.append((cursor, end, node_path or ["正文"], max(1, node.level)))
    return segments


def _book_title(book: NormalizedBook) -> str:
    for child in book.structure.root.children:
        if child.title and not child.title.startswith("[隐式"):
            return child.title
    return Path(book.source.path).stem


def tokenize_book(
    book: NormalizedBook,
    tokenizer: TokenizerAdapter,
    config: Config,
    *,
    seq_start: int,
) -> Tuple[List[TokenizedChunk], Dict[str, Any]]:
    """Tokenize one accepted book into ordered chunks with merged short spans."""

    max_tokens = int(config.get("l2.max_tokens", 30000))
    min_tokens = int(config.get("l2.min_tokens", 256))
    separators = [str(value) for value in config.get("l2.separators", ["\n\n", "\n", "。！？；…", ".!?;", "，,"])]
    eos_len = len(tokenizer.eos_ids)
    bin_size = int(config.get("l3.bin_size", 32700))
    if max_tokens + eos_len > bin_size:
        raise PackingError(
            f"max_tokens + eos_len ({max_tokens}+{eos_len}) exceeds bin_size {bin_size}"
        )

    title = _book_title(book)
    chunks: List[TokenizedChunk] = []
    local_seq = seq_start
    for start, end, chapter_path, chapter_level in _content_segments(book.structure.root, []):
        start = max(0, min(len(book.text), start))
        end = max(start, min(len(book.text), end))
        if not book.text[start:end].strip():
            continue
        parts = split_span(book.text, start, end, tokenizer, max_tokens, separators)
        for ids, part_start, part_end in parts:
            if not ids:
                continue
            chunks.append(
                TokenizedChunk(
                    chunk_id="",
                    seq=local_seq,
                    source_sha256=book.source.sha256,
                    book_title=title,
                    chapter_path=chapter_path,
                    chapter_level=chapter_level,
                    char_start=part_start,
                    char_end=part_end,
                    ids=ids,
                    n_tokens=len(ids),
                    flags=[],
                )
            )
            local_seq += 1

    before_tokens = sum(chunk.n_tokens for chunk in chunks)
    merged = merge_short_chunks(
        chunks,
        min_tokens=min_tokens,
        max_tokens=max_tokens,
        allow_cross_chapter=str(config.get("l2.merge_policy", "same_chapter")) != "none",
    )
    after_tokens = sum(chunk.n_tokens for chunk in merged)
    if before_tokens != after_tokens:
        raise PackingError("short-chunk merge changed token total")
    if sum(chunk.n_tokens for chunk in merged) < min_tokens:
        reason = "book_below_min_tokens_after_chunking"
        return [], {
            "reason": reason,
            "before_chunks": len(chunks),
            "after_chunks": 0,
            "tokens": after_tokens,
        }

    for index, chunk in enumerate(merged):
        chunk.seq = seq_start + index
        chunk.chunk_id = f"{chunk.source_sha256[:12]}_{chunk.seq:08d}"
        chunk.n_tokens = len(chunk.ids)
        if chunk.n_tokens > max_tokens:
            raise PackingError(f"chunk {chunk.chunk_id} exceeds max_tokens: {chunk.n_tokens}")

    audit = {
        "reason": "",
        "before_chunks": len(chunks),
        "after_chunks": len(merged),
        "tokens": after_tokens,
        "short_merge_count": len(chunks) - len(merged),
        "cross_chapter_merge_count": sum("cross_chapter_merge" in chunk.flags for chunk in merged),
    }
    return merged, audit


def _log_bucket(value: int) -> str:
    lower = 1
    while lower * 2 <= max(1, value):
        lower *= 2
    upper = lower * 2 - 1
    return f"{lower}-{upper}"


def run_l2(config: Config, tokenizer: TokenizerAdapter, stage_hash: str, *, force: bool = False) -> Dict[str, Any]:
    """Execute L2 for every accepted book and build a global chunk index."""

    config.ensure_directories()
    lock_path = config.dirs["state"] / "locks" / "l2.lock"
    with FileLock(lock_path, timeout_seconds=float(config.get("runtime.lock_timeout_seconds", 0))):
        l1_manifest = read_stage_manifest(config, "l1")
        meta_dir = config.dirs["state"] / "l2" / "books"
        index_dir = config.dirs["state"] / "l2" / "index"
        meta_dir.mkdir(parents=True, exist_ok=True)
        index_dir.mkdir(parents=True, exist_ok=True)
        global_index_path = config.dirs["chunks"] / "index.jsonl"
        global_index_path.parent.mkdir(parents=True, exist_ok=True)

        chunk_files: List[Dict[str, Any]] = []
        audit_records: List[Dict[str, Any]] = []
        metrics_rows: List[Dict[str, Any]] = []
        next_seq = 0
        total_tokens = 0
        total_chunks = 0
        dropped_chunks = 0

        with JSONLWriter(global_index_path) as global_writer:
            for item in sorted(l1_manifest.get("books", []), key=lambda value: value["sha256"]):
                sha = str(item["sha256"])
                meta_path = meta_dir / f"{sha}.json"
                per_book_index = index_dir / f"{sha}.jsonl"
                marker_ok = (not force) and stage_is_done(config, "l2", stage_hash, key=sha) and meta_path.exists()
                if marker_ok:
                    candidate_meta = read_json(meta_path)
                    candidate_chunk = config.work_dir / str(candidate_meta["chunk_file"]["path"])
                    marker_ok = candidate_chunk.exists() and per_book_index.exists()
                if marker_ok:
                    meta = read_json(meta_path)
                    for ref in read_jsonl(per_book_index):
                        global_writer.write(ref)
                    chunk_files.append(meta["chunk_file"])
                    metrics_rows.append(meta["metrics"])
                    total_tokens += int(meta["metrics"]["tokens"])
                    total_chunks += int(meta["metrics"]["chunks"])
                    dropped_chunks += int(meta["metrics"].get("dropped_chunks", 0))
                    next_seq = max(next_seq, int(meta["seq_end"]))
                    continue

                accepted_path = config.work_dir / str(item["path"])
                record = read_json(accepted_path)
                book = normalized_from_record(record, config.work_dir)
                chunks, audit_data = tokenize_book(book, tokenizer, config, seq_start=next_seq)
                chunk_file = config.dirs["chunks"] / f"{sha}.jsonl"
                refs: List[Dict[str, Any]] = []
                with JSONLWriter(chunk_file) as writer:
                    for chunk in chunks:
                        offset, byte_length = writer.write(asdict(chunk))
                        refs.append(
                            {
                                "chunk_id": chunk.chunk_id,
                                "seq": chunk.seq,
                                "source_sha256": chunk.source_sha256,
                                "file": config.relative_to_work(chunk_file),
                                "offset": offset,
                                "byte_length": byte_length,
                                "n_tokens": chunk.n_tokens,
                            }
                        )
                atomic_write_jsonl(per_book_index, refs)
                for ref in refs:
                    global_writer.write(ref)

                token_values = [chunk.n_tokens for chunk in chunks]
                row = {
                    "sha256": sha,
                    "chunks": len(chunks),
                    "tokens": sum(token_values),
                    "min_chunk_tokens": min(token_values) if token_values else 0,
                    "mean_chunk_tokens": sum(token_values) / max(1, len(token_values)),
                    "max_chunk_tokens": max(token_values) if token_values else 0,
                    "dropped_chunks": 0 if chunks else 1,
                    "short_merge_count": audit_data.get("short_merge_count", 0),
                    "cross_chapter_merge_count": audit_data.get("cross_chapter_merge_count", 0),
                    "reason": audit_data.get("reason", ""),
                }
                if not chunks:
                    dropped_chunks += 1
                seq_end = next_seq + len(chunks)
                chunk_meta = {
                    "sha256": sha,
                    "chunk_file": {
                        "sha256": sha,
                        "path": config.relative_to_work(chunk_file),
                        "index_path": config.relative_to_work(per_book_index),
                        "chunks": len(chunks),
                        "tokens": sum(token_values),
                    },
                    "seq_start": next_seq,
                    "seq_end": seq_end,
                    "metrics": row,
                }
                atomic_write_json(meta_path, chunk_meta, compact=True)
                write_stage_marker(config, "l2", stage_hash, key=sha, extra={"chunks": len(chunks)})
                audit = {
                    "sha256": sha,
                    "stage": "l2",
                    "decision": "accept" if chunks else "drop",
                    "reason": audit_data.get("reason", ""),
                    "score": 1.0 if chunks else 0.0,
                    "config_hash": stage_hash,
                    **row,
                }
                write_audit_record(config, "l2", sha, audit)
                audit_records.append(audit)
                chunk_files.append(chunk_meta["chunk_file"])
                metrics_rows.append(row)
                total_tokens += sum(token_values)
                total_chunks += len(chunks)
                next_seq = seq_end

        for item in sorted(l1_manifest.get("books", []), key=lambda value: value["sha256"]):
            sha = str(item["sha256"])
            audit_path = config.dirs["audit"] / "records" / "l2" / f"{sha}.json"
            if audit_path.exists():
                audit_records.append(read_json(audit_path))
        audit_records = sorted({str(record["sha256"]): record for record in audit_records}.values(), key=lambda item: item["sha256"])
        compile_audit(config, "l2")

        chunk_files.sort(key=lambda value: value["sha256"])
        manifest = {
            "stage": "l2",
            "stage_hash": stage_hash,
            "chunk_files": chunk_files,
            "generated_count": len(chunk_files),
            "total_chunks": total_chunks,
            "total_tokens": total_tokens,
            "seq_end": next_seq,
            "global_index": config.relative_to_work(global_index_path),
        }
        write_stage_manifest(config, "l2", manifest)

        hist = Counter(_log_bucket(row["max_chunk_tokens"]) for row in metrics_rows if row["chunks"])
        metrics = {
            "stage": "l2",
            "books": len(metrics_rows),
            "chunks": total_chunks,
            "tokens": total_tokens,
            "dropped_books": dropped_chunks,
            "short_merge_count": sum(int(row.get("short_merge_count", 0)) for row in metrics_rows),
            "cross_chapter_merge_count": sum(int(row.get("cross_chapter_merge_count", 0)) for row in metrics_rows),
            "chunk_length_histogram": dict(sorted(hist.items())),
        }
        atomic_write_json(config.dirs["metrics"] / "l2.json", metrics)
        write_stage_marker(config, "l2", stage_hash)
        return metrics
