"""Command-line interface for the TXT-to-JSONL pretraining pipeline."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Optional, Sequence

from ebook_pipeline.config import load_config
from ebook_pipeline.errors import ConfigError, PackingError, PipelineError, StateError, ValidationError
from ebook_pipeline.pipeline import Pipeline, STAGES
from ebook_pipeline.summary import summarize
from ebook_pipeline.tokenizer import load_tokenizer
from ebook_pipeline.validation import inspect_lineage, validate_chunks, validate_dataset


def _default_config() -> str:
    environment = os.environ.get("EBOOK_PIPELINE_CONFIG")
    if environment:
        return environment
    for candidate in ["config.yaml", "configs/production.yaml", "configs/dev.yaml"]:
        if Path(candidate).exists():
            return candidate
    return "configs/production.yaml"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ebook-pipeline",
        description="Deterministic TXT-to-JSONL dataset builder for LLM pretraining.",
    )
    parser.add_argument("--config", default=_default_config(), help="YAML configuration file")
    parser.add_argument("--profile", choices=["development", "production"], help="override profile")
    parser.add_argument("--log-level", default=None, help="DEBUG, INFO, WARNING, ERROR")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init = subparsers.add_parser("init", help="create configured directories without processing")
    init.add_argument("--books-dir", default=None)

    run = subparsers.add_parser("run", help="run L0 through L4 and optional validation")
    run.add_argument("--force", action="store_true", help="recompute stages even when markers match")
    run.add_argument("--no-validate", action="store_true", help="skip release validation")
    run.add_argument("--validate", action="store_true", help="force release validation")

    stage = subparsers.add_parser("stage", help="run one stage")
    stage.add_argument("stage", choices=STAGES)
    stage.add_argument("--force", action="store_true")
    stage.add_argument("--validate", action="store_true", help="run validation after L4")

    validate = subparsers.add_parser("validate", help="validate shards, offsets, and sampled chunks")
    validate.add_argument("--sample-packs", type=int, default=10)
    validate.add_argument("--sample-chunks", type=int, default=1000)

    jsonl_export = subparsers.add_parser(
        "export-jsonl", help="export cleaned text as {\"text\": ...} JSONL"
    )
    jsonl_export.add_argument("--output", default="jsonl/pretrain.jsonl")
    jsonl_export.add_argument("--max-tokens", type=int, required=True)
    jsonl_export.add_argument(
        "--tokenizer-path",
        default=None,
        help="training tokenizer directory or tokenizer.json used for exact length budgeting",
    )
    jsonl_export.add_argument("--force", action="store_true", help="recompute pipeline stages before export")
    lineage = subparsers.add_parser("lineage", help="trace one pack to source files and chapters")
    lineage.add_argument("pack_id")

    subparsers.add_parser("config-hash", help="print config and tokenizer fingerprint hashes")
    subparsers.add_parser("summary", help="rebuild metrics/summary.json")
    reproduce = subparsers.add_parser(
        "repro-check", help="rerun into a separate work dir and compare final shard bytes"
    )
    reproduce.add_argument("--alt-work-dir", required=True)
    return parser


def _print_json(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=False))


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level or "INFO").upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        config = load_config(args.config, profile=args.profile)
        if args.command == "init":
            config.ensure_directories()
            config.books_dir.mkdir(parents=True, exist_ok=True)
            if args.books_dir:
                Path(args.books_dir).mkdir(parents=True, exist_ok=True)
            _print_json({"work_dir": str(config.work_dir), "books_dir": str(config.books_dir), "status": "ready"})
            return 0

        tokenizer = load_tokenizer(config)
        fingerprint = tokenizer.fingerprint(config)
        if args.command == "config-hash":
            _print_json(
                {
                    "config_hash": config.config_hash(fingerprint),
                    "tokenizer_fingerprint": fingerprint,
                    "tokenizer_name": tokenizer.name,
                }
            )
            return 0


        if args.command == "lineage":
            _print_json(inspect_lineage(config, args.pack_id))
            return 0

        if args.command == "summary":
            _print_json(summarize(config, config.config_hash(fingerprint)))
            return 0

        if args.command == "repro-check":
            from ebook_pipeline.determinism import reproduce_and_compare

            result = reproduce_and_compare(config, Path(args.alt_work_dir))
            _print_json(result)
            return 0 if result["passed"] else 3

        if args.command in {"run", "stage", "validate", "export-jsonl"}:
            config.validate(require_runtime_assets=True)
            config.validate_assets()
        if args.command == "export-jsonl":
            pipeline = Pipeline(config, force=bool(args.force))
            pipeline.run_all(validate=False)
            from ebook_pipeline.jsonl_export import export_text_jsonl, load_training_tokenizer

            tokenizer_path_value = args.tokenizer_path or os.environ.get("TRAINING_TOKENIZER_PATH")
            export_tokenizer = (
                load_training_tokenizer(Path(tokenizer_path_value))
                if tokenizer_path_value
                else tokenizer
            )
            output = Path(args.output)
            if not output.is_absolute():
                output = config.work_dir / output
            _print_json(export_text_jsonl(config, export_tokenizer, output, max_tokens=args.max_tokens))
            return 0

        pipeline = Pipeline(config, force=bool(getattr(args, "force", False)))
        if args.command == "run":
            result = pipeline.run_all(validate=False if args.no_validate else (True if args.validate else None))
            _print_json(result)
            return 0
        if args.command == "stage":
            result = pipeline.run_stage(args.stage, validate=bool(args.validate))
            _print_json(result)
            return 0
        if args.command == "validate":
            result = validate_dataset(config, sample_packs=args.sample_packs)
            result["chunks"] = validate_chunks(config, tokenizer, sample_size=args.sample_chunks)
            _print_json(result)
            return 0
        parser.error(f"unsupported command: {args.command}")
        return 2
    except (ConfigError, StateError, PackingError, ValidationError, PipelineError) as exc:
        logging.error("%s", exc)
        return 2
    except Exception as exc:  # pragma: no cover - defensive CLI boundary
        logging.exception("unexpected failure: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
