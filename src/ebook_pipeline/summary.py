"""Funnel metrics and alert aggregation across all pipeline stages."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from ebook_pipeline.config import Config
from ebook_pipeline.io_utils import atomic_write_json, read_json


def _read_metric(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    value = read_json(path)
    return value if isinstance(value, dict) else {}


def summarize(config: Config, config_hash: str) -> Dict[str, Any]:
    """Build the final cross-stage summary and actionable alert flags."""

    metrics = {
        name: _read_metric(config.dirs["metrics"] / f"{name}.json")
        for name in ["l0", "l1", "l2", "l3", "l4"]
    }
    chunk_validation = _read_metric(config.dirs["metrics"] / "chunk_validation.json")
    l0, l1, l2, l3, l4 = (metrics[name] for name in ["l0", "l1", "l2", "l3", "l4"])
    input_count = int(l0.get("unique_documents", 0))
    accepted_count = int(l1.get("accepted", 0))
    rejection_rate = 1.0 - accepted_count / max(1, input_count)
    fill_ratio = float(l3.get("fill_ratio", 0.0))
    cross_doc_ratio = float(l3.get("cross_doc_ratio", 0.0))
    target_language_ratio = float(l1.get("target_language_token_ratio", 0.0))
    cross_doc_8gram_ratio = float(chunk_validation.get("cross_doc_8gram_repeat_ratio", 0.0))
    detector_distribution = l1.get("language_detector_distribution", {})
    fasttext_used = bool(l1.get("fasttext_used", False))

    rejection_threshold = float(config.get("runtime.alert_rejection_rate", 0.40))
    fill_threshold = float(config.get("runtime.alert_fill_ratio", 0.90))
    cross_doc_threshold = float(config.get("runtime.alert_cross_doc_ratio", 0.30))
    language_threshold = float(config.get("runtime.alert_target_language_ratio", 0.95))
    cross_doc_8gram_threshold = float(config.get("runtime.alert_cross_doc_8gram_ratio", 0.01))

    alerts = []
    if input_count and rejection_rate > rejection_threshold:
        alerts.append({"name": "rejection_rate", "value": rejection_rate, "threshold": rejection_threshold})
    if l3 and fill_ratio < fill_threshold:
        alerts.append({"name": "fill_ratio", "value": fill_ratio, "threshold": fill_threshold})
    if l3 and cross_doc_ratio > cross_doc_threshold:
        alerts.append({"name": "cross_doc_ratio", "value": cross_doc_ratio, "threshold": cross_doc_threshold})
    if l1 and target_language_ratio < language_threshold:
        alerts.append(
            {
                "name": "target_language_token_ratio",
                "value": target_language_ratio,
                "threshold": language_threshold,
            }
        )
    if config.profile == "production" and accepted_count and not fasttext_used:
        alerts.append(
            {
                "name": "language_detector_fallback",
                "value": detector_distribution,
                "threshold": "fasttext",
            }
        )
    if chunk_validation and cross_doc_8gram_ratio >= cross_doc_8gram_threshold:
        alerts.append(
            {
                "name": "cross_doc_8gram_repeat_ratio",
                "value": cross_doc_8gram_ratio,
                "threshold": cross_doc_8gram_threshold,
            }
        )

    summary = {
        "config_hash": config_hash,
        "funnel": {
            "input_documents": input_count,
            "l0_accepted": int(l0.get("accepted", 0)),
            "l1_accepted": accepted_count,
            "chunks": int(l2.get("chunks", 0)),
            "packs": int(l3.get("packs", 0)),
            "tokens": int(l4.get("tokens", 0)),
        },
        "fill_ratio": fill_ratio,
        "cross_doc_ratio": cross_doc_ratio,
        "rejection_rate": rejection_rate,
        "target_language_token_ratio": target_language_ratio,
        "cross_doc_8gram_repeat_ratio": cross_doc_8gram_ratio,
        "rejection_reasons": l1.get("rejection_reasons", {}),
        "language_distribution": l1.get("language_distribution", {}),
        "language_token_distribution": l1.get("language_token_distribution", {}),
        "language_detector_distribution": detector_distribution,
        "fasttext_used": fasttext_used,
        "quality_score_distribution": l1.get("score_distribution", {}),
        "chunk_length_histogram": l2.get("chunk_length_histogram", {}),
        "alerts": alerts,
        "status": "review_required" if alerts else "ok",
    }
    atomic_write_json(config.dirs["metrics"] / "summary.json", summary)
    return summary
