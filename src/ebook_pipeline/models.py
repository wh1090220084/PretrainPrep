"""Public data contracts for the TXT-to-JSONL preprocessing pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class SourceMeta:
    """Immutable identity and ingestion metadata for one source file."""

    path: str
    sha256: str
    size_bytes: int
    encoding_detected: str
    ingest_ts: str
    aliases: List[str] = field(default_factory=list)


@dataclass
class QualityVector:
    """Cheap statistical descriptors computed during L0/L1."""

    replacement_ratio: float
    control_ratio: float
    cjk_ratio: float
    latin_ratio: float
    punct_ratio: float
    dup_ngram_ratio: float
    url_density: float
    lang: str
    lang_conf: float


@dataclass
class StructureNode:
    """A node in the hierarchical chapter tree.

    Offsets are Unicode character offsets into the cleaned
    ``NormalizedBook.text``. The root node uses level ``-1`` and a synthetic
    title so every content byte has exactly one leaf owner.
    """

    level: int
    title: str
    char_start: int
    char_end: int
    children: List["StructureNode"] = field(default_factory=list)


@dataclass
class StructureTree:
    """Root wrapper for a chapter tree."""

    root: StructureNode


@dataclass
class NormalizedBook:
    """A decoded, cleaned document plus its structural model."""

    source: SourceMeta
    text: str
    quality: QualityVector
    structure: StructureTree
    flags: List[str] = field(default_factory=list)


@dataclass
class TokenizedChunk:
    """A contiguous tokenized span in one normalized document."""

    chunk_id: str
    seq: int
    source_sha256: str
    book_title: str
    chapter_path: List[str]
    chapter_level: int
    char_start: int
    char_end: int
    ids: List[int]
    n_tokens: int
    flags: List[str] = field(default_factory=list)


@dataclass
class PackedSequence:
    """A fixed-capacity training sequence containing one or more chunks."""

    pack_id: str
    input_ids: List[int]
    doc_lengths: List[int]
    position_ids: List[int]
    loss_mask: List[int]
    doc_ids: List[str]
    source_sha256s: List[str]
    fill_ratio: float
    cross_doc: bool


@dataclass
class GateResult:
    """Decision returned by a quality gate."""

    decision: str
    reason: str
    score: float
    stage: str
    details: Dict[str, Any] = field(default_factory=dict)




def _structure_to_dict(node: StructureNode) -> Dict[str, Any]:
    return {
        "level": node.level,
        "title": node.title,
        "char_start": node.char_start,
        "char_end": node.char_end,
        "children": [_structure_to_dict(child) for child in node.children],
    }


def _structure_from_dict(value: Dict[str, Any]) -> StructureNode:
    return StructureNode(
        level=int(value["level"]),
        title=str(value["title"]),
        char_start=int(value["char_start"]),
        char_end=int(value["char_end"]),
        children=[_structure_from_dict(child) for child in value.get("children", [])],
    )


def normalized_record(book: NormalizedBook, text_path: str) -> Dict[str, Any]:
    """Serialize a book without embedding its potentially huge text blob."""

    return {
        "source": asdict(book.source),
        "text_path": text_path,
        "quality": asdict(book.quality),
        "structure": _structure_to_dict(book.structure.root),
        "flags": list(book.flags),
    }


def normalized_from_record(record: Dict[str, Any], base_dir: Path) -> NormalizedBook:
    """Load a normalized record and the text file referenced by it."""

    source = SourceMeta(**record["source"])
    text_path = Path(record["text_path"])
    if not text_path.is_absolute():
        text_path = base_dir / text_path
    text = text_path.read_text(encoding="utf-8")
    quality = QualityVector(**record["quality"])
    structure = StructureTree(root=_structure_from_dict(record["structure"]))
    return NormalizedBook(
        source=source,
        text=text,
        quality=quality,
        structure=structure,
        flags=list(record.get("flags", [])),
    )

def get_nested(mapping: Dict[str, Any], key: str, default: Any = None) -> Any:
    """Read a dotted configuration key without raising KeyError."""

    current: Any = mapping
    for part in key.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current