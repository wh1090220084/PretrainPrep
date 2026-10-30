"""Minimal, dependency-free consumer for the final JSONL dataset.

Usage:
    python examples/consume_jsonl.py --dataset data/shards
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List

from ebook_pipeline.dataset import PackedDataset


def build_block_diagonal_mask(doc_lengths: List[int]) -> List[List[int]]:
    """Return a 0/1 mask that prevents attention across packed documents."""

    size = sum(doc_lengths)
    mask: List[List[int]] = [[0] * size for _ in range(size)]
    cursor = 0
    for length in doc_lengths:
        for row in range(cursor, cursor + length):
            for column in range(cursor, cursor + length):
                mask[row][column] = 1
        cursor += length
    return mask


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="directory containing index.jsonl and shard_*.jsonl")
    parser.add_argument("--pack-id", default=None)
    args = parser.parse_args()
    with PackedDataset(args.dataset) as dataset:
        pack_id = args.pack_id or str(dataset.rows[0]["pack_id"])
        pack: Dict[str, Any] = dataset[pack_id]
        mask = build_block_diagonal_mask([int(value) for value in pack["doc_lengths"]])
        summary = {
            "pack_id": pack_id,
            "n_tokens": len(pack["input_ids"]),
            "n_docs": len(pack["doc_ids"]),
            "doc_lengths": pack["doc_lengths"],
            "cross_doc": pack["cross_doc"],
            "source_sha256s": pack["source_sha256s"],
            "attention_mask_shape": [len(mask), len(mask[0]) if mask else 0],
            "first_document_ids": pack["input_ids"][:16],
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())