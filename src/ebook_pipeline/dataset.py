"""Training-side random access reader for final JSONL shards.

The index is loaded into memory because it contains one compact record per
pack. Token payloads remain on disk and are parsed only for requested packs.
"""

from __future__ import annotations

import contextlib
import json
from pathlib import Path
from typing import Any, BinaryIO, Dict, Iterator, List, Union

from ebook_pipeline.io_utils import iter_jsonl


class PackedDataset:
    """Random-access dataset over ``shards/index.jsonl``."""

    def __init__(self, dataset_dir: Union[str, Path]) -> None:
        self.dataset_dir = Path(dataset_dir).resolve()
        self.index_path = self.dataset_dir / "index.jsonl"
        self.manifest_path = self.dataset_dir / "manifest.json"
        if not self.index_path.exists():
            raise FileNotFoundError(f"missing shard index: {self.index_path}")
        self.rows: List[Dict[str, Any]] = list(iter_jsonl(self.index_path))
        self.by_id: Dict[str, Dict[str, Any]] = {str(row["pack_id"]): row for row in self.rows}
        if len(self.by_id) != len(self.rows):
            raise ValueError("duplicate pack_id in shard index")
        self._handles: Dict[Path, BinaryIO] = {}

    def __len__(self) -> int:
        return len(self.rows)

    def _handle(self, path: Path) -> BinaryIO:
        if path not in self._handles:
            self._handles[path] = path.open("rb")
        return self._handles[path]

    def __getitem__(self, pack_id: str) -> Dict[str, Any]:
        row = self.by_id[str(pack_id)]
        path = self.dataset_dir / str(row["file"])
        handle = self._handle(path)
        handle.seek(int(row["offset"]))
        payload = handle.read(int(row["byte_length"]))
        if len(payload) != int(row["byte_length"]):
            raise ValueError(f"short shard read for {pack_id}")
        value = json.loads(payload.decode("utf-8"))
        if value.get("pack_id") != pack_id:
            raise ValueError(f"pack index mismatch for {pack_id}")
        return value

    def rows_for_shard(self, shard_id: int) -> List[Dict[str, Any]]:
        return [row for row in self.rows if int(row["shard_id"]) == int(shard_id)]

    def close(self) -> None:
        for handle in self._handles.values():
            with contextlib.suppress(OSError):
                handle.close()
        self._handles.clear()

    def __enter__(self) -> "PackedDataset":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def iter_packs(self) -> Iterator[Dict[str, Any]]:
        for row in self.rows:
            yield self[str(row["pack_id"])]