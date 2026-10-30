"""L0 ingestion, decoding, cleaning, quality vectors, and chapter structure."""

from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from ebook_pipeline.config import Config
from ebook_pipeline.errors import DecodeError
from ebook_pipeline.io_utils import (
    FileLock,
    atomic_write_json,
    atomic_write_jsonl,
    atomic_write_text,
    compile_audit,
    sha256_file,
    stage_is_done,
    utc_now_iso,
    write_audit_record,
    write_quarantine,
    write_stage_manifest,
    write_stage_marker,
)
from ebook_pipeline.models import (
    NormalizedBook,
    QualityVector,
    SourceMeta,
    StructureNode,
    StructureTree,
    normalized_record,
)

PAGE_PATTERNS = [
    re.compile(r"^\s*[-—–]?\s*\d{1,4}\s*[-—–]?\s*$"),
    re.compile(r"^\s*第\s*\d{1,5}\s*页\s*$"),
    re.compile(r"^\s*Page\s+\d{1,5}\s*$", re.IGNORECASE),
]
TOC_TITLE_RE = re.compile(r"^\s*(目\s*录|contents?)\s*$", re.IGNORECASE)
TOC_ENTRY_RE = re.compile(r"^\s*.{1,100}?(?:[.…·]{2,}|\s{2,})\s*\d{1,4}\s*$")
SENTENCE_END_RE = re.compile(r"[。！？!?；;…][”’\"'）)\]]?\s*$")
CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
LATIN_RE = re.compile(r"[A-Za-z]")
URL_RE = re.compile(r"(?i)\b(?:https?://|www\.)[^\s<>()]+")
CONTROL_CATEGORIES = {"Cc", "Cf", "Cs", "Co", "Cn"}


def _sample_text(text: str, max_chars: int = 1_000_000) -> str:
    if len(text) <= max_chars:
        return text
    third = max_chars // 3
    middle_start = max(0, len(text) // 2 - third // 2)
    return text[:third] + text[middle_start : middle_start + third] + text[-third:]


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def compute_quality_vector(
    text: str,
    *,
    lang: str = "und",
    lang_conf: float = 0.0,
    ngram_size: int = 3,
    max_ngram_chars: int = 1_000_000,
) -> QualityVector:
    """Compute deterministic, platform-independent L0/L1 statistics."""

    length = max(1, len(text))
    replacement = _ratio(text.count("\ufffd"), length)
    control = _ratio(
        sum(1 for char in text if unicodedata.category(char) in CONTROL_CATEGORIES and char not in "\n\t"),
        length,
    )
    cjk = _ratio(len(CJK_RE.findall(text)), length)
    latin = _ratio(len(LATIN_RE.findall(text)), length)
    punct = _ratio(sum(1 for char in text if unicodedata.category(char).startswith("P")), length)
    url_chars = sum(len(match.group(0)) for match in URL_RE.finditer(text))
    url_density = _ratio(url_chars, length)

    sample = _sample_text(text, max_ngram_chars)
    grams = [sample[index : index + ngram_size] for index in range(max(0, len(sample) - ngram_size + 1))]
    if grams:
        counts = Counter(grams)
        duplicate_positions = sum(count - 1 for count in counts.values())
        dup_ngram = _ratio(duplicate_positions, len(grams))
    else:
        dup_ngram = 0.0

    return QualityVector(
        replacement_ratio=replacement,
        control_ratio=control,
        cjk_ratio=cjk,
        latin_ratio=latin,
        punct_ratio=punct,
        dup_ngram_ratio=dup_ngram,
        url_density=url_density,
        lang=lang,
        lang_conf=lang_conf,
    )


def _candidate_metrics(text: str) -> Tuple[float, float, float, float]:
    length = max(1, len(text))
    replacement = _ratio(text.count("\ufffd"), length)
    control = _ratio(
        sum(1 for char in text if unicodedata.category(char) in CONTROL_CATEGORIES and char not in "\n\t"),
        length,
    )
    nul = _ratio(text.count("\x00"), length)
    readable = _ratio(sum(1 for char in text if char.isprintable() or char in "\n\r\t"), length)
    return replacement, control, nul, readable


def detect_and_decode(
    data: bytes,
    *,
    replacement_ratio_max: float = 0.001,
    control_ratio_max: float = 0.02,
    whitelist: Optional[Sequence[str]] = None,
) -> Tuple[str, str]:
    """Decode bytes with BOM priority and deterministic candidate arbitration.

    Strict UTF-8 is always preferred when valid. Otherwise each candidate is
    decoded with replacement characters and scored by replacement/control/NUL
    rates plus printable density. This is intentionally deterministic and does
    not silently depend on a platform locale.
    """

    allowed = {item.lower().replace("_", "-") for item in (whitelist or ["utf-8", "gb18030", "big5", "utf-16"])}
    bom_candidates = [
        (b"\xef\xbb\xbf", "utf-8-sig", "utf-8"),
        (b"\xff\xfe\x00\x00", "utf-32-le", "utf-32"),
        (b"\x00\x00\xfe\xff", "utf-32-be", "utf-32"),
        (b"\xff\xfe", "utf-16-le", "utf-16"),
        (b"\xfe\xff", "utf-16-be", "utf-16"),
    ]
    for bom, codec, family in bom_candidates:
        if data.startswith(bom) and family in allowed:
            try:
                text = data.decode(codec)
            except UnicodeDecodeError:
                continue
            replacement, control, _nul, _readable = _candidate_metrics(text)
            if replacement <= replacement_ratio_max and control <= control_ratio_max:
                return text, codec

    try:
        text = data.decode("utf-8")
        return text, "utf-8"
    except UnicodeDecodeError:
        pass

    candidates: List[Tuple[str, str]] = []
    if "gb18030" in allowed:
        candidates.append(("gb18030", "gb18030"))
    if "big5" in allowed:
        candidates.append(("big5", "big5"))
    if "utf-16" in allowed:
        candidates.extend([("utf-16-le", "utf-16-le"), ("utf-16-be", "utf-16-be")])

    scored: List[Tuple[float, int, str, str]] = []
    for order, (codec, label) in enumerate(candidates):
        try:
            text = data.decode(codec, errors="replace")
            if text.startswith("\ufeff"):
                text = text[1:]
        except LookupError:
            continue
        replacement, control, nul, readable = _candidate_metrics(text)
        if replacement > replacement_ratio_max or control > control_ratio_max:
            continue
        # NULs strongly signal wrong endianness; printable density and
        # replacement cost dominate, with candidate order as a stable tie-break.
        score = readable - (2.5 * nul) - (1.5 * control) - (4.0 * replacement)
        scored.append((score, -order, label, text))

    if not scored:
        raise DecodeError("no configured encoding passed replacement/control gates")
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    _score, _order, label, text = scored[0]
    return text, label


def _is_heading_line(line: str, max_chars: int) -> bool:
    stripped = line.strip()
    if not stripped or len(stripped) > max_chars:
        return False
    if stripped.endswith(("。", "！", "？", "；", "…")):
        return False
    return True


def match_heading(line: str, *, conservative: bool, max_chars: int = 80) -> Optional[Tuple[int, str]]:
    """Recognize a conservative or loose heading and return ``(level, title)``."""

    if not _is_heading_line(line, max_chars):
        return None
    stripped = line.strip()
    markdown = re.match(r"^(#{1,3})\s+(.+?)\s*$", stripped)
    if markdown:
        return min(2, len(markdown.group(1)) - 1), markdown.group(2).strip()

    strict_patterns = [
        (0, r"^第[零一二三四五六七八九十百千万两〇0-9]+[卷部篇]\s*(.*)$"),
        (1, r"^第[零一二三四五六七八九十百千万两〇0-9]+[章回]\s*(.*)$"),
        (2, r"^第[零一二三四五六七八九十百千万两〇0-9]+节\s*(.*)$"),
        (1, r"^(?:Chapter|CHAPTER)\s+[0-9IVXLCDMivxlcdm]+\b\s*(.*)$"),
        (1, r"^(序章|序言|前言|楔子|引子|后记|尾声|附录|致谢)\s*[:：]?\s*(.*)$"),
    ]
    for level, pattern in strict_patterns:
        match = re.match(pattern, stripped)
        if match:
            title = next((group.strip() for group in match.groups() if group and group.strip()), stripped)
            return level, title
    if conservative:
        return None

    loose_patterns = [
        (2, r"^(\d{1,4})[.、]\s*(\S.{0,79})$"),
        (2, r"^([一二三四五六七八九十百千]+)、\s*(\S.{0,79})$"),
        (2, r"^(\d{1,4})\s+(\S.{0,79})$"),
    ]
    for level, pattern in loose_patterns:
        match = re.match(pattern, stripped)
        if match:
            return level, stripped
    return None


def _split_heading_records(text: str, cfg: Config) -> List[Tuple[int, int, int, str]]:
    max_chars = int(cfg.get("l0.structure.heading_max_chars", 80))
    radius = int(cfg.get("l0.structure.loose_scan_radius", 2))
    strict: List[Tuple[int, int, int, str]] = []
    loose: List[Tuple[int, int, int, str]] = []
    offset = 0
    lines = text.splitlines(keepends=True)
    for index, line in enumerate(lines):
        content = line.rstrip("\n")
        strict_match = match_heading(content, conservative=True, max_chars=max_chars)
        if strict_match:
            strict.append((offset, strict_match[0], index, strict_match[1]))
        else:
            loose_match = match_heading(content, conservative=False, max_chars=max_chars)
            if loose_match:
                loose.append((offset, loose_match[0], index, loose_match[1]))
        offset += len(line)

    strict_line_numbers = {item[2] for item in strict}
    merged = list(strict)
    for item in loose:
        line_number = item[2]
        if any(abs(line_number - existing) <= radius for existing in strict_line_numbers):
            continue
        merged.append(item)
    merged.sort(key=lambda item: (item[0], item[1], item[3]))
    return merged


def _assign_ends(records: List[Tuple[int, int, int, str]], text_length: int) -> List[int]:
    ends: List[int] = []
    for index, (_offset, level, _line, _title) in enumerate(records):
        end = text_length
        for later_offset, later_level, _later_line, _later_title in records[index + 1 :]:
            if later_level <= level:
                end = later_offset
                break
        ends.append(end)
    return ends


def detect_structure(text: str, cfg: Config) -> Tuple[StructureTree, List[str]]:
    """Identify conservative headings, arbitrate fallback, and build a tree."""

    flags: List[str] = []
    records = _split_heading_records(text, cfg)
    max_chapters = int(cfg.get("l0.structure.max_chapters", 300))
    avg_tokens_min = float(cfg.get("l0.structure.avg_tokens_min", 100))
    chars_per_token = float(cfg.get("l0.structure.chars_per_token", 2.0))

    if records:
        ends_for_estimate = _assign_ends(records, len(text))
        content_chars = sum(end - start for (start, _l, _line, _t), end in zip(records, ends_for_estimate))
        average_chars = content_chars / len(records)
    else:
        average_chars = float(len(text))

    if len(records) > max_chapters or (records and average_chars < avg_tokens_min * chars_per_token):
        flags.append("structure_fallback")
        root = StructureNode(level=-1, title="正文", char_start=0, char_end=len(text))
        leaf = StructureNode(level=1, title="正文", char_start=0, char_end=len(text))
        root.children.append(leaf)
        return StructureTree(root=root), flags

    ends = _assign_ends(records, len(text))
    root = StructureNode(level=-1, title="正文", char_start=0, char_end=len(text))
    stack: List[StructureNode] = [root]

    first_start = records[0][0] if records else len(text)
    if first_start > 0 and text[:first_start].strip():
        root.children.append(StructureNode(level=1, title="正文开头", char_start=0, char_end=first_start))

    for (start, level, _line, title), end in zip(records, ends):
        while stack[-1].level >= level:
            stack.pop()
        if level > stack[-1].level + 1:
            for implicit_level in range(stack[-1].level + 1, level):
                implicit = StructureNode(
                    level=implicit_level,
                    title=f"[隐式层级{implicit_level}]",
                    char_start=start,
                    char_end=end,
                )
                stack[-1].children.append(implicit)
                stack.append(implicit)
                flags.append("implicit_structure_level")
        node = StructureNode(level=level, title=title, char_start=start, char_end=end)
        stack[-1].children.append(node)
        stack.append(node)

    def update_parent_ends(node: StructureNode) -> None:
        if not node.children:
            return
        for child in node.children:
            update_parent_ends(child)
        node.char_end = max(node.char_end, max(child.char_end for child in node.children))

    update_parent_ends(root)
    if not root.children:
        root.children.append(StructureNode(level=1, title="正文", char_start=0, char_end=len(text)))
    return StructureTree(root=root), sorted(set(flags))


def clean_text(text: str, cfg: Config) -> Tuple[str, List[str], Dict[str, int]]:
    """Apply the ordered cleaning rules and return a normalized text document."""

    flags: List[str] = []
    deletions = Counter()
    had_carriage_return = "\r" in text
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if had_carriage_return:
        flags.append("newline_normalized")

    lines = text.split("\n")
    page_max = int(cfg.get("l0.cleaning.page_number_max_chars", 12))
    filtered: List[str] = []
    for line in lines:
        stripped = line.strip()
        if len(stripped) <= page_max and any(pattern.match(line) for pattern in PAGE_PATTERNS):
            deletions["page_lines"] += 1
            continue
        filtered.append(line)
    lines = filtered

    toc_scan = int(cfg.get("l0.cleaning.toc_scan_lines", 200))
    toc_start = None
    for index, line in enumerate(lines[:toc_scan]):
        if TOC_TITLE_RE.match(line):
            toc_start = index
            break
    if toc_start is not None:
        remove_until = toc_start + 1
        entries = 0
        cursor = toc_start + 1
        while cursor < len(lines) and cursor <= toc_start + toc_scan:
            stripped = lines[cursor].strip()
            if not stripped:
                remove_until = cursor + 1
                cursor += 1
                continue
            if TOC_ENTRY_RE.match(lines[cursor]):
                entries += 1
                remove_until = cursor + 1
                cursor += 1
                continue
            break
        if entries >= 2:
            del lines[toc_start:remove_until]
            deletions["toc_lines"] += remove_until - toc_start
            flags.append("toc_removed")

    total_lines = sum(1 for line in lines if line.strip())
    min_count = max(
        int(cfg.get("l0.cleaning.header_footer_min_count", 5)),
        int(math.ceil(total_lines * float(cfg.get("l0.cleaning.header_footer_min_ratio", 0.01)))),
    )
    max_chars = int(cfg.get("l0.cleaning.header_footer_max_chars", 40))
    counts = Counter(line.strip() for line in lines if line.strip())
    removable = {
        line
        for line, count in counts.items()
        if count >= min_count
        and len(line) <= max_chars
        and not SENTENCE_END_RE.search(line)
        and match_heading(line, conservative=True, max_chars=max_chars) is None
        and match_heading(line, conservative=False, max_chars=max_chars) is None
    }
    if removable:
        kept: List[str] = []
        for line in lines:
            if line.strip() in removable:
                deletions["header_footer_lines"] += 1
                continue
            kept.append(line)
        lines = kept
        flags.append("header_footer_removed")

    max_joins = int(cfg.get("l0.cleaning.hard_wrap_max_joins", 5))
    rebuilt: List[str] = []
    join_count = 0
    for line in lines:
        current = line
        if not rebuilt or not current.strip() or join_count >= max_joins:
            rebuilt.append(current)
            join_count = 0
            continue
        previous = rebuilt[-1]
        prev_stripped = previous.rstrip()
        cur_stripped = current.lstrip()
        cur_indented = len(current) - len(cur_stripped) > 0
        prev_is_heading = match_heading(prev_stripped, conservative=False, max_chars=80) is not None
        cur_is_heading = match_heading(cur_stripped, conservative=False, max_chars=80) is not None
        if (
            prev_stripped
            and not SENTENCE_END_RE.search(prev_stripped)
            and not prev_is_heading
            and not cur_indented
            and not cur_is_heading
        ):
            separator = " " if prev_stripped[-1:].isascii() and cur_stripped[:1].isascii() else ""
            rebuilt[-1] = prev_stripped + separator + cur_stripped
            join_count += 1
            deletions["hard_wrap_joins"] += 1
            continue
        rebuilt.append(current)
        join_count = 0

    text = "\n".join(rebuilt)
    before_blanks = text
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if text != before_blanks:
        flags.append("blank_lines_compressed")
    if any(step for step in deletions.values()):
        flags.append("cleaned")
    return text, sorted(set(flags)), dict(sorted(deletions.items()))


def _scan_books(books_dir: Path) -> List[Path]:
    if not books_dir.exists():
        return []
    return sorted(
        (path for path in books_dir.rglob("*") if path.is_file() and path.suffix.lower() == ".txt"),
        key=lambda path: path.relative_to(books_dir).as_posix(),
    )


def _top_reasons(records: Iterable[Dict[str, Any]], limit: int = 10) -> Dict[str, int]:
    counts = Counter(str(record.get("reason", "")) for record in records if record.get("decision") != "accept")
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:limit])


def _count_nodes(node: StructureNode) -> int:
    return 1 + sum(_count_nodes(child) for child in node.children)


def _leaf_nodes(node: StructureNode) -> List[StructureNode]:
    if not node.children:
        return [node]
    leaves: List[StructureNode] = []
    for child in node.children:
        leaves.extend(_leaf_nodes(child))
    return leaves


def run_l0(config: Config, stage_hash: str, *, force: bool = False) -> Dict[str, Any]:
    """Execute L0 and write normalized books, quarantine evidence, and manifests."""

    config.ensure_directories()
    lock_path = config.dirs["state"] / "locks" / "l0.lock"
    with FileLock(lock_path, timeout_seconds=float(config.get("runtime.lock_timeout_seconds", 0))):
        paths = _scan_books(config.books_dir)
        by_sha: Dict[str, List[Path]] = {}
        for path in paths:
            sha = sha256_file(path)
            by_sha.setdefault(sha, []).append(path)

        normalized_dir = config.dirs["normalized"]
        accepted_manifest: List[Dict[str, Any]] = []
        all_audit: List[Dict[str, Any]] = []
        input_records: List[Dict[str, Any]] = []
        rejected_count = 0
        total_chars = 0

        for sha in sorted(by_sha):
            group = sorted(by_sha[sha], key=lambda path: path.relative_to(config.books_dir).as_posix())
            canonical = group[0]
            aliases = [path.relative_to(config.books_dir).as_posix() for path in group[1:]]
            source_rel = canonical.relative_to(config.books_dir).as_posix()
            audit_path = config.dirs["audit"] / "records" / "l0" / f"{sha}.json"
            normalized_json = normalized_dir / f"{sha}.json"
            normalized_txt = normalized_dir / f"{sha}.txt"
            marker_ok = (not force) and stage_is_done(config, "l0", stage_hash, key=sha)

            reusable_audit = None
            if marker_ok and audit_path.exists():
                reusable_audit = json.loads(audit_path.read_text(encoding="utf-8"))
            if reusable_audit is not None and (
                reusable_audit.get("decision") != "accept" or (normalized_json.exists() and normalized_txt.exists())
            ):
                audit = reusable_audit
                if audit.get("decision") == "accept":
                    accepted_manifest.append(
                        {
                            "sha256": sha,
                            "path": config.relative_to_work(normalized_json),
                            "text_path": config.relative_to_work(normalized_txt),
                            "source_path": source_rel,
                            "aliases": aliases,
                        }
                    )
                    total_chars += int(audit.get("char_count", 0))
                else:
                    rejected_count += 1
            else:
                try:
                    data = canonical.read_bytes()
                    raw_text, encoding = detect_and_decode(
                        data,
                        replacement_ratio_max=float(config.get("l0.replacement_ratio_max", 0.001)),
                        control_ratio_max=float(config.get("l0.control_ratio_max", 0.005)),
                        whitelist=config.get("l0.encoding_whitelist", []),
                    )
                    cleaned, cleaning_flags, deletions = clean_text(raw_text, config)
                    quality = compute_quality_vector(cleaned)
                    if quality.replacement_ratio > float(config.get("l0.replacement_ratio_max", 0.001)):
                        raise DecodeError("replacement_ratio_after_cleaning")
                    if quality.control_ratio > float(config.get("l0.control_ratio_max", 0.005)):
                        raise DecodeError("control_ratio_after_cleaning")
                    if len(cleaned) < int(config.get("l0.min_chars", 1024)):
                        raise DecodeError("too_short")
                    structure, structure_flags = detect_structure(cleaned, config)
                    source = SourceMeta(
                        path=source_rel,
                        sha256=sha,
                        size_bytes=len(data),
                        encoding_detected=encoding,
                        ingest_ts=utc_now_iso(),
                        aliases=aliases,
                    )
                    book = NormalizedBook(
                        source=source,
                        text=cleaned,
                        quality=quality,
                        structure=structure,
                        flags=sorted(set(cleaning_flags + structure_flags)),
                    )
                    text_rel = config.relative_to_work(normalized_txt)
                    atomic_write_text(normalized_txt, cleaned)
                    record = normalized_record(book, text_rel)
                    atomic_write_json(normalized_json, record)
                    audit = {
                        "sha256": sha,
                        "stage": "l0",
                        "decision": "accept",
                        "reason": "",
                        "score": 1.0,
                        "config_hash": stage_hash,
                        "source_path": source_rel,
                        "aliases": aliases,
                        "encoding": encoding,
                        "char_count": len(cleaned),
                        "flags": book.flags,
                        "cleaning": deletions,
                        "quality": asdict(quality),
                        "structure": {
                            "root_title": structure.root.title,
                            "node_count": _count_nodes(structure.root),
                            "leaf_count": len(_leaf_nodes(structure.root)),
                        },
                    }
                    write_audit_record(config, "l0", sha, audit)
                    write_stage_marker(config, "l0", stage_hash, key=sha, extra={"decision": "accept"})
                    all_audit.append(audit)
                    accepted_manifest.append(
                        {
                            "sha256": sha,
                            "path": config.relative_to_work(normalized_json),
                            "text_path": text_rel,
                            "source_path": source_rel,
                            "aliases": aliases,
                        }
                    )
                    total_chars += len(cleaned)
                except (DecodeError, OSError, ValueError) as exc:
                    reason = str(exc) if isinstance(exc, DecodeError) else "ingest_error"
                    rejected_count += 1
                    audit = {
                        "sha256": sha,
                        "stage": "l0",
                        "decision": "quarantine",
                        "reason": reason,
                        "score": 0.0,
                        "config_hash": stage_hash,
                        "source_path": source_rel,
                        "aliases": aliases,
                    }
                    write_quarantine(config, "l0", sha, audit)
                    write_audit_record(config, "l0", sha, audit)
                    write_stage_marker(config, "l0", stage_hash, key=sha, extra={"decision": "quarantine"})
                    all_audit.append(audit)

            for alias in aliases:
                input_records.append(
                    {
                        "path": alias,
                        "sha256": sha,
                        "canonical_path": source_rel,
                        "decision": "duplicate_input",
                    }
                )
            input_records.append({"path": source_rel, "sha256": sha, "canonical_path": source_rel, "decision": "ingested"})

        input_records.sort(key=lambda record: record["path"])
        audit_records = sorted(all_audit, key=lambda record: record["sha256"])
        compile_audit(config, "l0")
        atomic_write_jsonl(config.dirs["audit"] / "l0_inputs.jsonl", input_records)
        manifest = {
            "stage": "l0",
            "stage_hash": stage_hash,
            "books": sorted(accepted_manifest, key=lambda item: item["sha256"]),
            "input_file_count": len(paths),
            "unique_sha_count": len(by_sha),
            "accepted_count": len(accepted_manifest),
            "quarantined_count": rejected_count,
            "total_chars": total_chars,
        }
        write_stage_manifest(config, "l0", manifest)
        metrics = {
            "stage": "l0",
            "input_files": len(paths),
            "unique_documents": len(by_sha),
            "accepted": len(accepted_manifest),
            "quarantined": rejected_count,
            "duplicate_input_files": len(paths) - len(by_sha),
            "total_chars": total_chars,
            "rejection_reasons": _top_reasons(audit_records),
            "acceptance_rate": _ratio(len(accepted_manifest), len(by_sha)),
        }
        atomic_write_json(config.dirs["metrics"] / "l0.json", metrics)
        write_stage_marker(config, "l0", stage_hash)
        return metrics
