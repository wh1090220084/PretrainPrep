"""Deterministic JSON/JSONL I/O, hashes, locks, manifests, and audit helpers."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import socket
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from ebook_pipeline.config import Config
from ebook_pipeline.errors import StateError


def utc_now_iso() -> str:
    """Return a timezone-aware, second-resolution UTC timestamp."""

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def stable_json_dumps(value: Any) -> str:
    """Serialize JSON deterministically while retaining dataclass field order."""

    return json.dumps(value, ensure_ascii=False, sort_keys=False, separators=(",", ":"))


def canonical_json(value: Any) -> str:
    """Serialize for hashing; mapping keys are sorted recursively."""

    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    """Hash a file without loading it into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    return sha256_text(canonical_json(value))


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _temporary_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write bytes via a same-directory temporary file and atomic replace."""

    ensure_parent(path)
    temporary = _temporary_path(path)
    try:
        with temporary.open("wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            with contextlib.suppress(OSError):
                temporary.unlink()


def atomic_write_text(path: Path, text: str, encoding: str = "utf-8") -> None:
    atomic_write_bytes(path, text.encode(encoding))


def atomic_write_json(path: Path, value: Any, *, compact: bool = False) -> None:
    if compact:
        text = stable_json_dumps(value) + "\n"
    else:
        text = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    atomic_write_text(path, text)


def atomic_write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    temporary = _temporary_path(path)
    ensure_parent(path)
    try:
        with temporary.open("wb") as handle:
            for record in records:
                handle.write(stable_json_dumps(record).encode("utf-8"))
                handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            with contextlib.suppress(OSError):
                temporary.unlink()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise StateError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise StateError(f"JSONL record at {path}:{line_number} is not an object")
            yield value


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return list(iter_jsonl(path))


class JSONLWriter:
    """Streaming JSONL writer that reports exact byte offsets.

    When ``atomic`` is true the destination is replaced only after a clean
    context-manager exit. This prevents a crashed writer from publishing a
    truncated shard.
    """

    def __init__(self, path: Path, *, atomic: bool = True) -> None:
        self.path = path
        self.atomic = atomic
        self._target = _temporary_path(path) if atomic else path
        self._handle = None
        self.offset = 0

    def __enter__(self) -> "JSONLWriter":
        ensure_parent(self.path)
        self._handle = self._target.open("wb")
        return self

    def write(self, record: Dict[str, Any]) -> Tuple[int, int]:
        if self._handle is None:
            raise RuntimeError("JSONLWriter is not open")
        payload = stable_json_dumps(record).encode("utf-8") + b"\n"
        offset = self.offset
        self._handle.write(payload)
        self.offset += len(payload)
        return offset, len(payload)

    def close(self, *, success: bool = True) -> None:
        if self._handle is None:
            return
        self._handle.flush()
        os.fsync(self._handle.fileno())
        self._handle.close()
        self._handle = None
        if self.atomic and success:
            os.replace(self._target, self.path)

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        success = exc_type is None
        self.close(success=success)
        if not success and self._target.exists():
            with contextlib.suppress(OSError):
                self._target.unlink()


def file_sha256_list(paths: Sequence[Path]) -> str:
    """Hash named files in caller-defined order, including their basenames."""

    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


class FileLock:
    """Small cross-platform O_EXCL lock for a single stage output directory."""

    def __init__(self, path: Path, *, timeout_seconds: float = 0.0) -> None:
        self.path = path
        self.timeout_seconds = max(0.0, timeout_seconds)

    def __enter__(self) -> "FileLock":
        ensure_parent(self.path)
        deadline = time.monotonic() + self.timeout_seconds
        while True:
            try:
                descriptor = os.open(str(self.path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                payload = json.dumps(
                    {
                        "pid": os.getpid(),
                        "host": socket.gethostname(),
                        "created_at": utc_now_iso(),
                    }
                ).encode("utf-8")
                os.write(descriptor, payload)
                os.close(descriptor)
                return self
            except FileExistsError:
                if time.monotonic() >= deadline:
                    raise StateError(
                        f"stage lock exists: {self.path}; remove it only after confirming no process is running"
                    )
                time.sleep(0.2)

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        with contextlib.suppress(FileNotFoundError):
            self.path.unlink()


def stage_marker_path(config: Config, stage: str, key: str = "run") -> Path:
    safe_key = key.replace("/", "_").replace("\\", "_")
    return config.dirs["state"] / stage / f"{safe_key}.done"


def write_stage_marker(
    config: Config,
    stage: str,
    stage_hash: str,
    *,
    key: str = "run",
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    path = stage_marker_path(config, stage, key)
    payload: Dict[str, Any] = {
        "stage": stage,
        "key": key,
        "stage_hash": stage_hash,
        "completed_at": utc_now_iso(),
    }
    if extra:
        payload.update(extra)
    atomic_write_json(path, payload, compact=True)
    return path


def stage_is_done(config: Config, stage: str, stage_hash: str, *, key: str = "run") -> bool:
    path = stage_marker_path(config, stage, key)
    if not path.exists():
        return False
    try:
        marker = read_json(path)
    except (OSError, ValueError):
        return False
    return marker.get("stage_hash") == stage_hash


def manifest_path(config: Config, stage: str) -> Path:
    return config.dirs["state"] / stage / "manifest.json"


def write_stage_manifest(config: Config, stage: str, payload: Dict[str, Any]) -> Path:
    path = manifest_path(config, stage)
    atomic_write_json(path, payload)
    return path


def read_stage_manifest(config: Config, stage: str) -> Dict[str, Any]:
    path = manifest_path(config, stage)
    if not path.exists():
        raise StateError(f"stage manifest missing: {path}")
    value = read_json(path)
    if not isinstance(value, dict):
        raise StateError(f"invalid stage manifest: {path}")
    return value


def write_audit_record(config: Config, stage: str, sha256: str, record: Dict[str, Any]) -> None:
    path = config.dirs["audit"] / "records" / stage / f"{sha256}.json"
    atomic_write_json(path, record, compact=True)


def compile_audit(config: Config, stage: str) -> Path:
    """Compile per-document audit records into one deterministic JSONL file."""

    records_dir = config.dirs["audit"] / "records" / stage
    output = config.dirs["audit"] / f"{stage}.jsonl"
    records: List[Dict[str, Any]] = []

    def key(path: Path) -> Tuple[str, str]:
        try:
            value = read_json(path)
            return (str(value.get("sha256", "")), path.name)
        except (OSError, ValueError):
            return ("", path.name)

    if records_dir.exists():
        for path in sorted(records_dir.glob("*.json"), key=key):
            records.append(read_json(path))
    atomic_write_jsonl(output, records)
    return output


def quarantine_path(config: Config, stage: str, sha256: str, suffix: str = ".json") -> Path:
    return config.dirs["quarantine"] / stage / f"{sha256}{suffix}"


def write_quarantine(
    config: Config,
    stage: str,
    sha256: str,
    payload: Dict[str, Any],
    *,
    suffix: str = ".json",
) -> Path:
    path = quarantine_path(config, stage, sha256, suffix)
    atomic_write_json(path, payload)
    return path
