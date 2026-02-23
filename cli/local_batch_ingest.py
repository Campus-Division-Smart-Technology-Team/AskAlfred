#!/usr/bin/env python3
"""
CLI entrypoint for AskAlfred local batch ingestion.
"""

from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alfred_exceptions import ConfigError, UnexpectedError
from config import BatchIngestConfig, NAMESPACE_MAPPINGS
from ingest import validate_namespace_routing, ingest_local_directory_with_progress, IngestContext
from dotenv import load_dotenv
import argparse
import logging


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Ingest local documents into Pinecone via OpenAI embeddings"
    )
    parser.add_argument("--path", help="Local directory path")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip documents that already exist in the index",
    )
    parser.add_argument(
        "--force-reindex",
        action="store_true",
        help="Force re-indexing of all documents (overrides skip-existing)",
    )
    parser.add_argument(
        "--io-workers",
        type=int,
        help="Number of IO workers for processing",
    )
    parser.add_argument(
        "--parse-workers",
        type=int,
        help="Number of parse workers for FRA extraction",
    )
    parser.add_argument(
        "--validate-routing",
        action="store_true",
        help="Run namespace routing validation tests",
    )
    parser.add_argument(
        "--export-events",
        action="store_true",
        help="Write building assignment events to JSONL file",
    )
    parser.add_argument(
        "--events-file",
        help="Path to JSONL export file for building assignment events",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bar display",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse + chunk only. Do NOT call OpenAI or Pinecone.",
    )
    parser.add_argument(
        "--upsert-strategy",
        choices=["worker", "inline"],
        help="Upsert strategy: background worker thread or inline.",
    )
    parser.add_argument(
        "--upsert-workers",
        type=int,
        help="Number of upsert worker threads (worker strategy only).",
    )

    return parser.parse_args()


def main() -> int:
    load_dotenv()
    args = parse_args()

    try:
        config = BatchIngestConfig.from_env()

        if args.path:
            config.local_path = args.path
        if args.io_workers:
            config.max_io_workers = args.io_workers
        if args.parse_workers:
            config.max_parse_workers = args.parse_workers
        if args.force_reindex:
            config.skip_existing = False
        elif args.skip_existing:
            config.skip_existing = True
        if args.export_events:
            config.export_events = True
        if args.events_file:
            config.export_events_file = args.events_file
        if args.dry_run:
            config.dry_run = True
            config.skip_existing = False
        if args.upsert_strategy:
            config.upsert_strategy = args.upsert_strategy
        if args.upsert_workers is not None:
            config.upsert_workers = args.upsert_workers

        config.validate()

    except ConfigError as error:
        logging.error("Configuration error: %s", error)
        return 1
    except UnexpectedError as error:
        logging.error("Configuration error: %s", error)
        return 1

    if args.validate_routing:
        for doc_type, expected_namespace in NAMESPACE_MAPPINGS.items():
            valid, reason = validate_namespace_routing(
                doc_type, expected_namespace)
            if not valid:
                raise ValueError(
                    f"Routing validation failed for doc_type='{doc_type}': {reason}"
                )
        logging.info("Namespace routing validation passed.")
        return 0

    ctx = IngestContext(config)

    try:
        ingest_local_directory_with_progress(
            ctx, use_progress_bar=not args.no_progress)
        return 0
    except KeyboardInterrupt:
        ctx.logger.warning("Ingestion interrupted by user. Cleaning up...")
        ctx.logger.info("No cache to persist on shutdown.")
        return 0
    except UnexpectedError as error:
        ctx.logger.error("Ingestion failed: %s", error, exc_info=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
