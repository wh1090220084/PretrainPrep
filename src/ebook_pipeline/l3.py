"""L3 OBFD packing with a capacity Fenwick tree and deterministic JSONL output."""

from __future__ import annotations

import contextlib
import heapq
import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Deque, Dict, Iterator, List, Optional, Sequence, Tuple

from ebook_pipeline.config import Config
from ebook_pipeline.errors import PackingError
from ebook_pipeline.io_utils import (
    FileLock,
    JSONLWriter,
    atomic_write_json,
    atomic_write_jsonl,
    iter_jsonl,
    read_json,
    read_stage_manifest,
    stage_is_done,
    write_stage_manifest,
    write_stage_marker,
)
from ebook_pipeline.models import PackedSequence


class ChunkReader:
    """Seek to JSONL byte offsets without loading the full chunk corpus."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self._handles: Dict[Path, Any] = {}

    def _handle(self, path: Path) -> Any:
        if path not in self._handles:
            self._handles[path] = path.open("rb")
        return self._handles[path]

    def read(self, ref: Dict[str, Any]) -> Dict[str, Any]:
        path = self.config.work_dir / str(ref["file"])
        handle = self._handle(path)
        handle.seek(int(ref["offset"]))
        line = handle.readline(int(ref["byte_length"]))
        if len(line) != int(ref["byte_length"]):
            raise PackingError(f"short JSONL read for {ref['chunk_id']} in {path}")
        value = json.loads(line.decode("utf-8"))
        if value.get("chunk_id") != ref.get("chunk_id"):
            raise PackingError(f"chunk index mismatch: {ref.get('chunk_id')} != {value.get('chunk_id')}")
        if len(value.get("ids", [])) != int(ref["n_tokens"]):
            raise PackingError(f"chunk token count mismatch: {ref['chunk_id']}")
        return value

    def close(self) -> None:
        for handle in self._handles.values():
            with contextlib.suppress(OSError):
                handle.close()
        self._handles.clear()

    def __enter__(self) -> "ChunkReader":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()


class FenwickCapacity:
    """Counts occupied capacities and supports first capacity >= target."""

    def __init__(self, max_capacity: int) -> None:
        self.max_capacity = max_capacity
        self.size = max_capacity + 1
        self.tree = [0] * (self.size + 1)

    def add(self, capacity: int, delta: int) -> None:
        index = capacity + 1
        while index <= self.size:
            self.tree[index] += delta
            index += index & -index

    def prefix(self, capacity: int) -> int:
        """Return the count of bins with capacity <= ``capacity``."""

        if capacity < 0:
            return 0
        index = min(self.max_capacity, capacity) + 1
        total = 0
        while index > 0:
            total += self.tree[index]
            index -= index & -index
        return total

    def total(self) -> int:
        return self.prefix(self.max_capacity)

    def kth(self, order: int) -> int:
        """Return the capacity of the 1-based k-th occupied bin."""

        if order <= 0 or order > self.total():
            raise IndexError("Fenwick order out of range")
        index = 0
        bit = 1 << (self.size.bit_length() - 1)
        while bit:
            next_index = index + bit
            if next_index <= self.size and self.tree[next_index] < order:
                index = next_index
                order -= self.tree[next_index]
            bit >>= 1
        return index

    def first_at_least(self, target: int) -> Optional[int]:
        before = self.prefix(target - 1)
        if self.total() <= before:
            return None
        return self.kth(before + 1)


class BFDPacker:
    """Best-fit decreasing bin assignment over exact remaining capacities."""

    def __init__(self, bin_size: int, *, per_chunk_overhead: int = 0) -> None:
        self.bin_size = bin_size
        self.per_chunk_overhead = per_chunk_overhead
        self.capacity_index = FenwickCapacity(bin_size)
        self.queues: List[Deque[int]] = [deque() for _ in range(bin_size + 1)]
        self.bins: List[List[Dict[str, Any]]] = []

    def add(self, ref: Dict[str, Any]) -> int:
        size = int(ref["n_tokens"]) + self.per_chunk_overhead
        if size <= 0 or size > self.bin_size:
            raise PackingError(f"chunk {ref.get('chunk_id')} has invalid packed size {size}")
        capacity = self.capacity_index.first_at_least(size)
        if capacity is None:
            bin_index = len(self.bins)
            self.bins.append([])
            remaining = self.bin_size - size
        else:
            queue = self.queues[capacity]
            if not queue:
                raise PackingError("capacity index and bin queues diverged")
            bin_index = queue.popleft()
            self.capacity_index.add(capacity, -1)
            remaining = capacity - size
        self.bins[bin_index].append(ref)
        self.queues[remaining].append(bin_index)
        self.capacity_index.add(remaining, 1)
        return bin_index


def _ref_from_record(record: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "chunk_id": str(record["chunk_id"]),
        "seq": int(record["seq"]),
        "source_sha256": str(record["source_sha256"]),
        "file": str(record["file"]),
        "offset": int(record["offset"]),
        "byte_length": int(record["byte_length"]),
        "n_tokens": int(record["n_tokens"]),
    }


def _sort_key(ref: Dict[str, Any]) -> Tuple[int, str, int]:
    return (-int(ref["n_tokens"]), str(ref["chunk_id"]), int(ref["seq"]))


@contextlib.contextmanager
def iter_sorted_refs(
    index_path: Path,
    temp_dir: Path,
    *,
    max_in_memory: int,
    sort_runs: int,
) -> Iterator[Iterator[Dict[str, Any]]]:
    """Return sorted refs in memory or through deterministic external merge sort."""

    count = 0
    for _record in iter_jsonl(index_path):
        count += 1
    if count <= max_in_memory:
        refs = [_ref_from_record(record) for record in iter_jsonl(index_path)]
        refs.sort(key=_sort_key)
        yield iter(refs)
        return

    temp_dir.mkdir(parents=True, exist_ok=True)
    run_paths: List[Path] = []
    run_size = max(1000, (count + max(1, sort_runs) - 1) // max(1, sort_runs))
    buffer: List[Dict[str, Any]] = []
    run_index = 0

    def flush() -> None:
        nonlocal buffer, run_index
        if not buffer:
            return
        buffer.sort(key=_sort_key)
        path = temp_dir / f"run_{run_index:05d}.jsonl"
        atomic_write_jsonl(path, buffer)
        run_paths.append(path)
        run_index += 1
        buffer = []

    try:
        for record in iter_jsonl(index_path):
            buffer.append(_ref_from_record(record))
            if len(buffer) >= run_size:
                flush()
        flush()
        iterators = [iter_jsonl(path) for path in run_paths]
        merged = heapq.merge(*iterators, key=lambda value: _sort_key(_ref_from_record(value)))
        yield (_ref_from_record(record) for record in merged)
    finally:
        for path in run_paths:
            with contextlib.suppress(OSError):
                path.unlink()
        with contextlib.suppress(OSError):
            temp_dir.rmdir()


def _build_pack(
    pack_id: str,
    refs: Sequence[Dict[str, Any]],
    reader: ChunkReader,
    *,
    eos_ids: Sequence[int],
    eos_loss: int,
    bin_size: int,
) -> PackedSequence:
    ordered = sorted(refs, key=lambda value: (int(value["seq"]), str(value["chunk_id"])))
    input_ids: List[int] = []
    doc_lengths: List[int] = []
    position_ids: List[int] = []
    loss_mask: List[int] = []
    doc_ids: List[str] = []
    source_shas: List[str] = []

    for ref in ordered:
        chunk = reader.read(ref)
        ids = [int(value) for value in chunk["ids"]]
        if len(ids) != int(ref["n_tokens"]):
            raise PackingError(f"token count drift for {ref['chunk_id']}")
        complete = ids + [int(value) for value in eos_ids]
        doc_length = len(complete)
        input_ids.extend(complete)
        doc_lengths.append(doc_length)
        position_ids.extend(range(doc_length))
        loss_mask.extend([1] * len(ids) + [int(eos_loss)] * len(eos_ids))
        doc_ids.append(str(chunk["chunk_id"]))
        source_shas.append(str(chunk["source_sha256"]))

    if len(input_ids) > bin_size:
        raise PackingError(f"{pack_id} overflow: {len(input_ids)} > {bin_size}")
    if sum(doc_lengths) != len(input_ids):
        raise PackingError(f"{pack_id} doc_lengths sum mismatch")
    if len(position_ids) != len(input_ids) or len(loss_mask) != len(input_ids):
        raise PackingError(f"{pack_id} sidecar length mismatch")

    unique_sources = sorted(set(source_shas))
    return PackedSequence(
        pack_id=pack_id,
        input_ids=input_ids,
        doc_lengths=doc_lengths,
        position_ids=position_ids,
        loss_mask=loss_mask,
        doc_ids=doc_ids,
        source_sha256s=unique_sources,
        fill_ratio=len(input_ids) / float(bin_size),
        cross_doc=len(unique_sources) > 1,
    )


def _group_by_book(refs: Sequence[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for ref in refs:
        groups[str(ref["source_sha256"])].append(ref)
    return [groups[key] for key in sorted(groups)]


def _write_packed_partition(
    partition_id: int,
    bins: Sequence[Sequence[Dict[str, Any]]],
    reader: ChunkReader,
    config: Config,
    *,
    eos_ids: Sequence[int],
    eos_loss: int,
    bin_size: int,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[PackedSequence]]:
    packed_path = config.dirs["packed"] / f"packed_{partition_id:05d}.jsonl"
    index_records: List[Dict[str, Any]] = []
    sequences: List[PackedSequence] = []
    with JSONLWriter(packed_path) as writer:
        for bin_index, refs in enumerate(bins):
            pack_id = f"pack_p{partition_id:05d}_{bin_index:08d}"
            sequence = _build_pack(
                pack_id,
                refs,
                reader,
                eos_ids=eos_ids,
                eos_loss=eos_loss,
                bin_size=bin_size,
            )
            from dataclasses import asdict

            offset, byte_length = writer.write(asdict(sequence))
            index_records.append(
                {
                    "pack_id": pack_id,
                    "file": config.relative_to_work(packed_path),
                    "offset": offset,
                    "byte_length": byte_length,
                    "n_tokens": len(sequence.input_ids),
                    "n_docs": len(sequence.doc_ids),
                    "cross_doc": sequence.cross_doc,
                    "source_shas": sequence.source_sha256s,
                    "fill_ratio": sequence.fill_ratio,
                }
            )
            sequences.append(sequence)
    return (
        {
            "partition": partition_id,
            "path": config.relative_to_work(packed_path),
            "packs": len(bins),
            "tokens": sum(len(sequence.input_ids) for sequence in sequences),
        },
        index_records,
        sequences,
    )


def _write_error(config: Config, exc: Exception) -> None:
    atomic_write_json(
        config.dirs["errors"] / "l3_overflow.json",
        {"stage": "l3", "error": type(exc).__name__, "message": str(exc)},
    )


def run_l3(config: Config, tokenizer: Any, stage_hash: str, *, force: bool = False) -> Dict[str, Any]:
    """Execute deterministic OBFD packing and emit indexed packed JSONL."""

    config.ensure_directories()
    lock_path = config.dirs["state"] / "locks" / "l3.lock"
    with FileLock(lock_path, timeout_seconds=float(config.get("runtime.lock_timeout_seconds", 0))):
        metrics_path = config.dirs["metrics"] / "l3.json"
        if not force and stage_is_done(config, "l3", stage_hash) and metrics_path.exists():
            return read_json(metrics_path)
        try:
            l2_manifest = read_stage_manifest(config, "l2")
            index_path = config.work_dir / str(l2_manifest["global_index"])
            bin_size = int(config.get("l3.bin_size", 32700))
            max_in_memory = int(config.get("l3.max_chunks_in_memory", 2_000_000))
            sort_runs = int(config.get("l3.external_sort_runs", 64))
            pack_mode = str(config.get("l3.pack_mode", "bfd"))
            eos_ids = tokenizer.eos_ids
            eos_loss = int(config.get("tokenizer.eos_loss", 1))
            temp_dir = config.get("l3.temp_dir")
            temp_path = config.resolve_path(str(temp_dir)) if temp_dir else config.dirs["state"] / "l3" / "sort_tmp"

            global_index_path = config.dirs["packed"] / "index.jsonl"
            partition_manifests: List[Dict[str, Any]] = []
            total_packs = 0
            total_tokens = 0
            cross_doc_packs = 0
            partition_id = 0

            with ChunkReader(config) as reader, JSONLWriter(global_index_path) as index_writer:
                with iter_sorted_refs(
                    index_path,
                    temp_path,
                    max_in_memory=max_in_memory,
                    sort_runs=sort_runs,
                ) as sorted_refs:
                    if pack_mode == "per_book":
                        refs = list(sorted_refs)
                        if len(refs) > max_in_memory:
                            raise PackingError(
                                "per_book mode requires the chunk corpus to fit max_chunks_in_memory"
                            )
                        groups = _group_by_book(refs)
                        for group in groups:
                            packer = BFDPacker(bin_size, per_chunk_overhead=len(eos_ids))
                            for ref in sorted(group, key=_sort_key):
                                packer.add(ref)
                            partition_manifest, index_records, sequences = _write_packed_partition(
                                partition_id,
                                packer.bins,
                                reader,
                                config,
                                eos_ids=eos_ids,
                                eos_loss=eos_loss,
                                bin_size=bin_size,
                            )
                            for record in index_records:
                                index_writer.write(record)
                            partition_manifests.append(partition_manifest)
                            total_packs += len(sequences)
                            total_tokens += sum(len(sequence.input_ids) for sequence in sequences)
                            cross_doc_packs += sum(sequence.cross_doc for sequence in sequences)
                            partition_id += 1
                    else:
                        partition: List[Dict[str, Any]] = []
                        for ref in sorted_refs:
                            partition.append(ref)
                            if len(partition) >= max_in_memory:
                                packer = BFDPacker(bin_size, per_chunk_overhead=len(eos_ids))
                                for item in partition:
                                    packer.add(item)
                                partition_manifest, index_records, sequences = _write_packed_partition(
                                    partition_id,
                                    packer.bins,
                                    reader,
                                    config,
                                    eos_ids=eos_ids,
                                    eos_loss=eos_loss,
                                    bin_size=bin_size,
                                )
                                for record in index_records:
                                    index_writer.write(record)
                                partition_manifests.append(partition_manifest)
                                total_packs += len(sequences)
                                total_tokens += sum(len(sequence.input_ids) for sequence in sequences)
                                cross_doc_packs += sum(sequence.cross_doc for sequence in sequences)
                                partition_id += 1
                                partition = []
                        if partition:
                            packer = BFDPacker(bin_size, per_chunk_overhead=len(eos_ids))
                            for item in partition:
                                packer.add(item)
                            partition_manifest, index_records, sequences = _write_packed_partition(
                                partition_id,
                                packer.bins,
                                reader,
                                config,
                                eos_ids=eos_ids,
                                eos_loss=eos_loss,
                                bin_size=bin_size,
                            )
                            for record in index_records:
                                index_writer.write(record)
                            partition_manifests.append(partition_manifest)
                            total_packs += len(sequences)
                            total_tokens += sum(len(sequence.input_ids) for sequence in sequences)
                            cross_doc_packs += sum(sequence.cross_doc for sequence in sequences)
                            partition_id += 1

            fill_ratio = total_tokens / float(max(1, total_packs * bin_size))
            manifest = {
                "stage": "l3",
                "stage_hash": stage_hash,
                "algorithm": "obfd",
                "pack_mode": pack_mode,
                "bin_size": bin_size,
                "packed_files": partition_manifests,
                "index": config.relative_to_work(global_index_path),
                "pack_count": total_packs,
                "total_tokens": total_tokens,
                "cross_doc_packs": cross_doc_packs,
            }
            write_stage_manifest(config, "l3", manifest)
            metrics = {
                "stage": "l3",
                "algorithm": "obfd",
                "pack_mode": pack_mode,
                "bin_size": bin_size,
                "packs": total_packs,
                "tokens": total_tokens,
                "fill_ratio": fill_ratio,
                "cross_doc_packs": cross_doc_packs,
                "cross_doc_ratio": cross_doc_packs / max(1, total_packs),
                "truncation_rate": 0.0,
                "partitions": len(partition_manifests),
            }
            atomic_write_json(config.dirs["metrics"] / "l3.json", metrics)
            write_stage_marker(config, "l3", stage_hash)
            atomic_write_jsonl(
                config.dirs["audit"] / "l3.jsonl",
                [
                    {
                        "stage": "l3",
                        "decision": "accept",
                        "reason": "",
                        "score": fill_ratio,
                        "config_hash": stage_hash,
                        "pack_count": total_packs,
                        "total_tokens": total_tokens,
                        "cross_doc_packs": cross_doc_packs,
                    }
                ],
            )
            return metrics
        except PackingError as exc:
            _write_error(config, exc)
            raise
