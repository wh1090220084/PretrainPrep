"""L1 language detection, quality gates, exact dedup, and SimHash-LSH dedup."""

from __future__ import annotations

import hashlib
import logging
import re
from collections import Counter, defaultdict
from dataclasses import asdict, replace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


LOGGER = logging.getLogger(__name__)
from ebook_pipeline.config import Config
from ebook_pipeline.errors import ConfigError
from ebook_pipeline.io_utils import (
    FileLock,
    atomic_write_json,
    atomic_write_jsonl,
    compile_audit,
    read_json,
    stage_is_done,
    write_audit_record,
    write_quarantine,
    write_stage_manifest,
    write_stage_marker,
)
from ebook_pipeline.models import GateResult, NormalizedBook, QualityVector, normalized_from_record
from ebook_pipeline.tokenizer import TokenizerAdapter

CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
LATIN_RE = re.compile(r"[A-Za-z]")
KANA_RE = re.compile(r"[\u3040-\u30ff]")
HANGUL_RE = re.compile(r"[\uac00-\ud7af]")
CYRILLIC_RE = re.compile(r"[\u0400-\u04ff]")
ARABIC_RE = re.compile(r"[\u0600-\u06ff]")
WORD_RE = re.compile(r"[A-Za-z]+|[\u3400-\u4dbf\u4e00-\u9fff]|\d+")
SENTENCE_SPLIT_RE = re.compile(r"[。！？!?；;\n]+")
COPYRIGHT_RE = re.compile(r"(?i)(copyright|all rights reserved|版权所有|版权声明|未经许可|不得转载)")
JUNK_LINE_RE = re.compile(r"^[\W_]+$")
TARGET_LANG_ALIASES = {"zh-cn": "zh", "zh-tw": "zh", "en-us": "en", "en-gb": "en"}


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


class LanguageDetector:
    """Language detection with fastText, auto fallback, and heuristic modes."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.requested_detector = str(config.get("l1.language.detector", "heuristic"))
        self.detector_name = self.requested_detector
        self.model_path = config.get("l1.language.model_path")
        self.heuristic_fallback = bool(config.get("l1.language.heuristic_fallback", False))
        self.max_chars = int(config.get("l1.language.max_chars", 5000))
        self.fallback_reason = ""
        self._model = None

        if self.requested_detector not in {"fasttext", "auto", "heuristic"}:
            raise ConfigError(f"unsupported language detector: {self.requested_detector}")
        if self.requested_detector == "heuristic":
            return

        try:
            import fasttext
        except ImportError:
            self.fallback_reason = "fasttext package is not installed"
        else:
            if not self.model_path:
                self.fallback_reason = "l1.language.model_path is not configured"
            else:
                path = config.resolve_path(str(self.model_path))
                if not path.exists():
                    self.fallback_reason = f"fastText model not found: {path}"
                else:
                    try:
                        self._model = fasttext.load_model(str(path))
                    except Exception as exc:
                        self.fallback_reason = f"failed to load fastText model: {exc}"

        if self._model is None:
            if self.requested_detector == "fasttext" and not self.heuristic_fallback:
                raise ConfigError(self.fallback_reason or "fastText detector is unavailable")
            self.detector_name = "heuristic"
            LOGGER.warning("Language detector fallback to heuristic: %s", self.fallback_reason)
        else:
            self.detector_name = "fasttext"

    def detect(self, text: str) -> Tuple[str, float, str]:
        sample = text[: self.max_chars].replace("\n", " ").strip()
        if self._model is not None and sample:
            try:
                labels, scores = self._model.predict(sample, k=1)
                label = str(labels[0]).replace("__label__", "").lower()
                label = TARGET_LANG_ALIASES.get(label, label)
                return label, float(scores[0]), "fasttext"
            except Exception as exc:
                self.fallback_reason = f"fastText prediction failed: {exc}"
                self._model = None
                self.detector_name = "heuristic"
                LOGGER.warning("Language detector fallback to heuristic: %s", self.fallback_reason)
        lang, conf = self._heuristic(sample)
        return lang, conf, "heuristic"


    @staticmethod
    def _heuristic(text: str) -> Tuple[str, float]:
        counts = {
            "zh": len(CJK_RE.findall(text)),
            "ja": len(KANA_RE.findall(text)),
            "ko": len(HANGUL_RE.findall(text)),
            "en": len(LATIN_RE.findall(text)),
            "ru": len(CYRILLIC_RE.findall(text)),
            "ar": len(ARABIC_RE.findall(text)),
        }
        if counts["ja"] >= 3:
            counts["zh"] = max(0, counts["zh"] - counts["ja"] // 2)
        total = sum(counts.values())
        if total == 0:
            return "und", 0.0
        lang, count = max(counts.items(), key=lambda item: (item[1], item[0]))
        return lang, count / total


class QualityGate:
    """Rule, statistical, and language gates with explicit rejection evidence."""

    def __init__(self, config: Config, tokenizer: TokenizerAdapter, detector: LanguageDetector) -> None:
        self.config = config
        self.tokenizer = tokenizer
        self.detector = detector
        self.gates = [str(value) for value in config.get("l1.gates", [])]
        self.ad_patterns = [re.compile(pattern) for pattern in config.get("l1.ad_patterns", [])]
        self.weights = {str(key): float(value) for key, value in config.get("l1.stat_weights", {}).items()}

    def evaluate(self, book: NormalizedBook) -> Tuple[GateResult, Dict[str, Any]]:
        text = book.text
        token_count = len(self.tokenizer.encode(text))
        lang, lang_conf, detector_backend = self.detector.detect(text)
        quality = replace(book.quality, lang=lang, lang_conf=lang_conf)
        diagnostics: Dict[str, Any] = {
            "token_count": token_count,
            "language": lang,
            "language_confidence": lang_conf,
            "language_detector": detector_backend,
            "language_detector_requested": self.detector.requested_detector,
            "language_detector_fallback_reason": self.detector.fallback_reason,
        }

        if any(name.startswith("rule") for name in self.gates):
            rule = self._rule_gate(text, token_count)
            if rule is not None:
                rule.details.update(diagnostics)
                return rule, diagnostics
        if any(name.startswith("lang") for name in self.gates):
            target = {str(value).lower() for value in self.config.get("l1.target_langs", ["zh", "en"])}
            min_conf = float(self.config.get("l1.language.min_confidence", 0.50))
            normalized_lang = TARGET_LANG_ALIASES.get(lang, lang)
            if normalized_lang not in target or lang_conf < min_conf:
                result = GateResult(
                    decision="reject",
                    reason="language_not_target",
                    score=lang_conf,
                    stage="l1.lang",
                    details={"language": lang, "target": sorted(target), **diagnostics},
                )
                return result, diagnostics

        stat_score = 1.0
        components: Dict[str, float] = {}
        if any(name.startswith("stat") for name in self.gates):
            stat_score, components = self._stat_score(text, quality)
            diagnostics["stat_components"] = components
            diagnostics["stat_score"] = stat_score
            threshold = float(self.config.get("l1.stat_threshold", 0.35))
            if stat_score < threshold:
                result = GateResult(
                    decision="reject",
                    reason="stat_score_below_threshold",
                    score=stat_score,
                    stage="l1.stat",
                    details={"threshold": threshold, **diagnostics},
                )
                return result, diagnostics

        return (
            GateResult(
                decision="accept",
                reason="",
                score=stat_score,
                stage="l1",
                details=diagnostics,
            ),
            diagnostics,
        )

    def _rule_gate(self, text: str, token_count: int) -> Optional[GateResult]:
        char_count = len(text)
        if char_count < int(self.config.get("l0.min_chars", 1024)):
            return GateResult("reject", "too_short_chars", 0.0, "l1.rule", {"char_count": char_count})
        min_tokens = int(self.config.get("l1.min_tokens", 256))
        if token_count < min_tokens:
            return GateResult(
                "reject", "too_short_tokens", token_count / max(1, min_tokens), "l1.rule", {"token_count": token_count}
            )

        url_matches = re.findall(r"(?i)\b(?:https?://|www\.)[^\s<>()]+", text)
        url_density = sum(len(item) for item in url_matches) / max(1, char_count)
        if url_density > float(self.config.get("l1.max_url_density", 0.002)):
            return GateResult("reject", "url_density", url_density, "l1.rule", {"url_density": url_density})

        for pattern in self.ad_patterns:
            match = pattern.search(text)
            if match:
                return GateResult(
                    "reject",
                    "ad_template",
                    float(match.end() - match.start()) / max(1, char_count),
                    "l1.rule",
                    {"matched_pattern": pattern.pattern, "match": match.group(0)[:120]},
                )

        lines = [line.strip() for line in text.splitlines() if line.strip()]
        junk_lines = sum(1 for line in lines if JUNK_LINE_RE.match(line))
        junk_ratio = junk_lines / max(1, len(lines))
        if junk_ratio > float(self.config.get("l1.max_junk_line_ratio", 0.10)):
            return GateResult("reject", "junk_line_ratio", junk_ratio, "l1.rule", {"junk_line_ratio": junk_ratio})

        max_copyright = int(self.config.get("l1.max_copyright_block_chars", 500))
        run_chars = 0
        for line in lines:
            if COPYRIGHT_RE.search(line):
                run_chars += len(line)
                if run_chars > max_copyright:
                    return GateResult(
                        "reject",
                        "copyright_block_too_long",
                        min(1.0, run_chars / max(1, max_copyright)),
                        "l1.rule",
                        {"copyright_run_chars": run_chars},
                    )
            else:
                run_chars = 0
        return None

    def _stat_score(self, text: str, quality: QualityVector) -> Tuple[float, Dict[str, float]]:
        sample = text[:500_000]
        words = WORD_RE.findall(sample)
        vocab_density = len(set(words)) / max(1, len(words))
        sentences = [part.strip() for part in SENTENCE_SPLIT_RE.split(sample) if part.strip()]
        avg_sentence_chars = sum(len(part) for part in sentences) / max(1, len(sentences))

        repetition = _clamp(1.0 - quality.dup_ngram_ratio / 0.45)
        lexical = _clamp((vocab_density - 0.05) / 0.35)
        if 8.0 <= avg_sentence_chars <= 120.0:
            sentence = 1.0
        elif avg_sentence_chars < 8.0:
            sentence = _clamp(avg_sentence_chars / 8.0)
        else:
            sentence = _clamp(1.0 - (avg_sentence_chars - 120.0) / 280.0)
        script = _clamp((quality.cjk_ratio + quality.latin_ratio) / 0.65)
        language = _clamp(quality.lang_conf)

        components = {
            "repetition": repetition,
            "lexical": lexical,
            "sentence": sentence,
            "script": script,
            "language": language,
        }
        denominator = sum(self.weights.get(name, 0.0) for name in components) or 1.0
        score = sum(self.weights.get(name, 0.0) * value for name, value in components.items()) / denominator
        components["vocab_density"] = vocab_density
        components["avg_sentence_chars"] = avg_sentence_chars
        return _clamp(score), components


def _blake2_128(value: str) -> int:
    return int.from_bytes(hashlib.blake2b(value.encode("utf-8"), digest_size=16).digest(), "big")


def _sample_shingles(text: str, shingle_size: int, max_shingles: int) -> List[str]:
    total = len(text) - shingle_size + 1
    if total <= 0:
        return [text] if text else []
    if total <= max_shingles:
        return [text[index : index + shingle_size] for index in range(total)]
    # Deterministic uniform sampling without random state.
    step = total / float(max_shingles)
    return [text[int(index * step) : int(index * step) + shingle_size] for index in range(max_shingles)]


_BYTE_BIT_LANES = []
for _byte_value in range(256):
    _packed = 0
    for _lane in range(8):
        if _byte_value & (1 << _lane):
            _packed |= 1 << (_lane * 32)
    _BYTE_BIT_LANES.append(_packed)


def simhash_128(
    text: str,
    *,
    bits: int = 128,
    shingle_size: int = 4,
    max_shingles: int = 500_000,
) -> Tuple[int, int]:
    """Compute a deterministic SimHash and return ``(value, shingles)``.

    Each 32-bit lane of a Python integer stores the positive vote count for one
    bit position. This reduces the hot loop from one update per bit to one
    update per hash byte while preserving the standard majority rule.
    """

    shingles = _sample_shingles(text, shingle_size, max_shingles)
    if not shingles:
        return 0, 0
    byte_count = (bits + 7) // 8
    counters = [0] * byte_count
    for shingle in shingles:
        h = _blake2_128(shingle)
        for byte_index in range(byte_count):
            counters[byte_index] += _BYTE_BIT_LANES[(h >> (byte_index * 8)) & 0xFF]
    threshold = len(shingles)
    value = 0
    for byte_index, packed in enumerate(counters):
        for lane in range(8):
            bit = byte_index * 8 + lane
            if bit >= bits:
                break
            count = (packed >> (lane * 32)) & 0xFFFFFFFF
            if count * 2 >= threshold:
                value |= 1 << bit
    return value, len(shingles)


def hamming_distance(left: int, right: int) -> int:
    value = left ^ right
    try:
        return int(value.bit_count())
    except AttributeError:
        return bin(value).count("1")


class UnionFind:
    def __init__(self) -> None:
        self.parent: Dict[str, str] = {}

    def add(self, value: str) -> None:
        self.parent.setdefault(value, value)

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        self.add(left)
        self.add(right)
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left == root_right:
            return
        if root_left < root_right:
            self.parent[root_right] = root_left
        else:
            self.parent[root_left] = root_right


def cluster_simhash(
    candidates: Sequence[Dict[str, Any]],
    *,
    hamming: int,
    bands: int,
    bits: int = 128,
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Cluster candidates by LSH buckets and return group/kept assignments."""

    union = UnionFind()
    buckets: Dict[Tuple[int, int], List[str]] = defaultdict(list)
    by_sha = {str(candidate["sha256"]): candidate for candidate in candidates}
    band_bits = bits // bands
    mask = (1 << band_bits) - 1

    for candidate in sorted(candidates, key=lambda item: item["sha256"]):
        sha = str(candidate["sha256"])
        value = int(candidate["simhash"], 16)
        union.add(sha)
        possible: Set[str] = set()
        for band in range(bands):
            key = (band, (value >> (band * band_bits)) & mask)
            possible.update(buckets[key])
        for other in sorted(possible, key=lambda item: (len(by_sha[item]["simhash"]), item)):
            other_value = int(by_sha[other]["simhash"], 16)
            if hamming_distance(value, other_value) <= hamming:
                union.union(sha, other)
        for band in range(bands):
            key = (band, (value >> (band * band_bits)) & mask)
            buckets[key].append(sha)

    groups: Dict[str, List[str]] = defaultdict(list)
    for sha in sorted(union.parent):
        groups[union.find(sha)].append(sha)
    kept: Dict[str, str] = {}
    group_ids: Dict[str, str] = {}
    for index, root in enumerate(sorted(groups)):
        members = sorted(groups[root])
        group_id = f"g{index:06d}"
        winner = min(
            members,
            key=lambda sha: (
                -int(by_sha[sha].get("shingle_count", 0)),
                -float(by_sha[sha].get("gate_score", 0.0)),
                sha,
            ),
        )
        for sha in members:
            group_ids[sha] = group_id
            kept[sha] = winner
    return group_ids, kept


def _top_counts(values: Iterable[str], limit: int = 10) -> Dict[str, int]:
    counts = Counter(values)
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:limit])


def _score_histogram(values: Sequence[float]) -> Dict[str, int]:
    buckets = {f"{index / 10:.1f}-{(index + 1) / 10:.1f}": 0 for index in range(10)}
    for value in values:
        index = min(9, max(0, int(float(value) * 10)))
        key = f"{index / 10:.1f}-{(index + 1) / 10:.1f}"
        buckets[key] += 1
    return buckets


def run_l1(config: Config, tokenizer: TokenizerAdapter, stage_hash: str, *, force: bool = False) -> Dict[str, Any]:
    """Execute quality gates and deterministic document deduplication."""

    config.ensure_directories()
    lock_path = config.dirs["state"] / "locks" / "l1.lock"
    with FileLock(lock_path, timeout_seconds=float(config.get("runtime.lock_timeout_seconds", 0))):
        from ebook_pipeline.io_utils import read_stage_manifest

        l0_manifest = read_stage_manifest(config, "l0")
        detector = LanguageDetector(config)
        gate = QualityGate(config, tokenizer, detector)
        candidate_dir = config.dirs["state"] / "l1" / "candidates"
        candidate_dir.mkdir(parents=True, exist_ok=True)
        candidates: List[Dict[str, Any]] = []

        for item in sorted(l0_manifest.get("books", []), key=lambda value: value["sha256"]):
            sha = str(item["sha256"])
            candidate_path = candidate_dir / f"{sha}.json"
            marker_ok = (not force) and stage_is_done(config, "l1", stage_hash, key=sha) and candidate_path.exists()
            if marker_ok:
                candidates.append(read_json(candidate_path))
                continue

            normalized_path = config.work_dir / str(item["path"])
            record = read_json(normalized_path)
            book = normalized_from_record(record, config.work_dir)
            result, diagnostics = gate.evaluate(book)
            candidate: Dict[str, Any] = {
                "sha256": sha,
                "normalized_path": str(item["path"]),
                "text_path": str(item["text_path"]),
                "source_path": book.source.path,
                "decision": result.decision,
                "reason": result.reason,
                "gate_stage": result.stage,
                "gate_score": result.score,
                "details": result.details,
                "diagnostics": diagnostics,
                "aliases": list(book.source.aliases),
                "simhash": None,
                "shingle_count": 0,
            }
            if result.decision == "accept":
                simhash, shingle_count = simhash_128(
                    book.text,
                    bits=int(config.get("l1.simhash_bits", 128)),
                    shingle_size=int(config.get("l1.simhash_shingle_size", 4)),
                    max_shingles=int(config.get("l1.simhash_max_shingles", 500_000)),
                )
                candidate["simhash"] = f"{simhash:032x}"
                candidate["shingle_count"] = shingle_count
            atomic_write_json(candidate_path, candidate, compact=True)
            write_stage_marker(config, "l1", stage_hash, key=sha, extra={"decision": result.decision})
            candidates.append(candidate)

        accepted_gate = [candidate for candidate in candidates if candidate["decision"] == "accept"]
        accepted_gate_by_sha = {str(candidate["sha256"]): candidate for candidate in accepted_gate}
        cluster_ids, kept_map = cluster_simhash(
            accepted_gate,
            hamming=int(config.get("l1.simhash_hamming", 3)),
            bands=int(config.get("l1.simhash_bands", 8)),
            bits=int(config.get("l1.simhash_bits", 128)),
        )

        accepted_manifest: List[Dict[str, Any]] = []
        dedup_index: List[Dict[str, Any]] = []
        audit_records: List[Dict[str, Any]] = []
        exact_duplicate_inputs = 0

        alias_duplicate_records: List[Dict[str, Any]] = []
        for candidate in sorted(candidates, key=lambda value: value["sha256"]):
            sha = str(candidate["sha256"])
            exact_duplicate_inputs += len(candidate.get("aliases", []))
            for alias_index, alias in enumerate(candidate.get("aliases", [])):
                alias_record = {
                    "sha256": sha,
                    "stage": "l1",
                    "decision": "reject",
                    "reason": "exact_dup",
                    "score": 1.0,
                    "config_hash": stage_hash,
                    "kept_sha": sha,
                    "source_path": alias,
                    "alias_index": alias_index,
                }
                write_quarantine(config, "l1", sha, alias_record, suffix=f".alias{alias_index}.json")
                alias_duplicate_records.append(alias_record)
            if candidate["decision"] != "accept":
                audit = {
                    "sha256": sha,
                    "stage": "l1",
                    "decision": "reject",
                    "reason": candidate["reason"],
                    "score": candidate["gate_score"],
                    "stage_detail": candidate["gate_stage"],
                    "config_hash": stage_hash,
                    "details": candidate.get("details", {}),
                }
                write_quarantine(config, "l1", sha, audit)
                write_audit_record(config, "l1", sha, audit)
                audit_records.append(audit)
                dedup_index.append(
                    {
                        "sha256": sha,
                        "simhash": candidate.get("simhash"),
                        "dup_group": None,
                        "kept": False,
                        "kept_sha": None,
                        "reason": candidate["reason"],
                    }
                )
                continue

            winner = kept_map[sha]
            group_id = cluster_ids[sha]
            if winner != sha:
                similarity = 1.0 - hamming_distance(
                    int(str(candidate["simhash"]), 16),
                    int(str(accepted_gate_by_sha[winner]["simhash"]), 16),
                ) / float(config.get("l1.simhash_bits", 128))
                audit = {
                    "sha256": sha,
                    "stage": "l1",
                    "decision": "reject",
                    "reason": "approximate_dup",
                    "score": similarity,
                    "config_hash": stage_hash,
                    "kept_sha": winner,
                    "dup_group": group_id,
                }
                write_quarantine(config, "l1", sha, audit)
                write_audit_record(config, "l1", sha, audit)
                audit_records.append(audit)
                dedup_index.append(
                    {
                        "sha256": sha,
                        "simhash": candidate["simhash"],
                        "dup_group": group_id,
                        "kept": False,
                        "kept_sha": winner,
                        "reason": "approximate_dup",
                    }
                )
                continue

            record = read_json(config.work_dir / str(candidate["normalized_path"]))
            language = candidate["diagnostics"].get("language", "und")
            language_confidence = float(candidate["diagnostics"].get("language_confidence", 0.0))
            quality = QualityVector(**record["quality"])
            record["quality"] = asdict(replace(quality, lang=language, lang_conf=language_confidence))
            record["l1"] = {
                "score": candidate["gate_score"],
                "simhash": candidate["simhash"],
                "dup_group": group_id,
                "details": candidate.get("details", {}),
            }
            accepted_path = config.dirs["accepted"] / f"{sha}.json"
            atomic_write_json(accepted_path, record)
            audit = {
                "sha256": sha,
                "stage": "l1",
                "decision": "accept",
                "reason": "",
                "score": candidate["gate_score"],
                "config_hash": stage_hash,
                "simhash": candidate["simhash"],
                "dup_group": group_id,
                "aliases": candidate.get("aliases", []),
            }
            write_audit_record(config, "l1", sha, audit)
            audit_records.append(audit)
            dedup_index.append(
                {
                    "sha256": sha,
                    "simhash": candidate["simhash"],
                    "dup_group": group_id,
                    "kept": True,
                    "kept_sha": sha,
                    "reason": "",
                }
            )
            accepted_manifest.append(
                {
                    "sha256": sha,
                    "path": config.relative_to_work(accepted_path),
                    "text_path": str(candidate["text_path"]),
                    "source_path": candidate["source_path"],
                    "dup_group": group_id,
                }
            )

        for alias_record in alias_duplicate_records:
            alias_record["kept_sha"] = kept_map.get(alias_record["sha256"], alias_record["kept_sha"])
            write_quarantine(
                config,
                "l1",
                alias_record["sha256"],
                alias_record,
                suffix=f".alias{alias_record['alias_index']}.json",
            )
        atomic_write_jsonl(config.dirs["audit"] / "l1_alias_duplicates.jsonl", alias_duplicate_records)
        dedup_index.sort(key=lambda value: value["sha256"])
        audit_records.sort(key=lambda value: value["sha256"])
        atomic_write_jsonl(config.dirs["dedup"] / "index.jsonl", dedup_index)
        compile_audit(config, "l1")

        gate_rejects = sum(1 for candidate in candidates if candidate["decision"] != "accept")
        dedup_rejects = len(accepted_gate) - len(accepted_manifest)
        manifest = {
            "stage": "l1",
            "stage_hash": stage_hash,
            "books": sorted(accepted_manifest, key=lambda item: item["sha256"]),
            "input_count": len(candidates),
            "accepted_count": len(accepted_manifest),
            "gate_rejected_count": gate_rejects,
            "dedup_rejected_count": dedup_rejects,
            "exact_duplicate_input_count": exact_duplicate_inputs,
        }
        write_stage_manifest(config, "l1", manifest)

        reasons = [str(record["reason"]) for record in audit_records if record["decision"] != "accept"]
        languages = [str(candidate["diagnostics"].get("language", "und")) for candidate in candidates]
        language_detectors = [
            str(candidate["diagnostics"].get("language_detector", "unknown")) for candidate in candidates
        ]
        detector_distribution = _top_counts(language_detectors, limit=10)
        fasttext_used = int(detector_distribution.get("fasttext", 0)) > 0
        target_langs = {TARGET_LANG_ALIASES.get(str(value).lower(), str(value).lower()) for value in config.get("l1.target_langs", ["zh", "en"])}
        accepted_shas = {str(item["sha256"]) for item in accepted_manifest}
        accepted_candidates = [candidate for candidate in candidates if str(candidate["sha256"]) in accepted_shas]
        language_tokens = Counter()
        for candidate in accepted_candidates:
            language = TARGET_LANG_ALIASES.get(
                str(candidate.get("diagnostics", {}).get("language", "und")),
                str(candidate.get("diagnostics", {}).get("language", "und")),
            )
            language_tokens[language] += int(candidate.get("diagnostics", {}).get("token_count", 0))
        total_language_tokens = sum(language_tokens.values())
        target_language_tokens = sum(count for lang, count in language_tokens.items() if lang in target_langs)
        target_language_token_ratio = target_language_tokens / max(1, total_language_tokens)
        metrics = {
            "stage": "l1",
            "input": len(candidates),
            "accepted": len(accepted_manifest),
            "gate_rejected": gate_rejects,
            "dedup_rejected": dedup_rejects,
            "exact_duplicate_input_files": exact_duplicate_inputs,
            "rejection_reasons": _top_counts(reasons),
            "language_distribution": _top_counts(languages, limit=20),
            "score_distribution": _score_histogram([float(c.get("gate_score", 0.0)) for c in candidates]),
            "acceptance_rate": len(accepted_manifest) / max(1, len(candidates)),
            "language_token_distribution": dict(sorted(language_tokens.items())),
            "target_language_token_ratio": target_language_token_ratio,
            "language_detector_distribution": detector_distribution,
            "fasttext_used": fasttext_used,
        }
        atomic_write_json(config.dirs["metrics"] / "l1.json", metrics)
        write_stage_marker(config, "l1", stage_hash)
        return metrics
