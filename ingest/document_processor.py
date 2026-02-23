#!/usr/bin/env python3
"""
Document processing pipeline.
This module defines the DocumentProcessor class, which orchestrates the processing of a single document from raw bytes to vector creation. 
It handles text extraction, chunking, embedding, and metadata enrichment, as well as special handling for fire risk assessments. 
The processor interacts with the FileRegistry for tracking file processing state and uses the BuildingResolver for building name resolution. 
It also includes logic for dry-run mode and event emission for building assignments.
"""

from __future__ import annotations

import hashlib
import uuid
import json
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from concurrent.futures import ProcessPoolExecutor
import time
from typing import TYPE_CHECKING, Any, Protocol
from redis.exceptions import RedisError
from config import (DocumentTypes, _route_namespace,
                    INGEST_LOW_CONFIDENCE_WARN, INGEST_EMBED_BATCH_SIZE,
                    INGEST_PROCESSING_LEASE_SECONDS,)
from alfred_exceptions import (
    ExternalServiceError,
    IngestError,
    RoutingError,
    ValidationError,
)
from filename_building_parser import should_flag_for_review
from fra import FraVectorExtractResult, extract_fra_metadata
from .document_content import (
    chunk_text_generator,
    chunk_text,
    embed_texts_batch,
    ext,
    extract_maintenance_csv,
    extract_text,
    extract_text_csv_by_building_enhanced,
    fetch_bytes_secure,
    is_bms_document,
    is_fire_risk_assessment,
    extract_text_and_chunk_in_process,
)
from .utils import (
    enrich_with_building_metadata,
    validate_with_truncation,
    validate_namespace_routing,
)

if TYPE_CHECKING:
    from .context import IngestContext
    from building.resolver import BuildingResolver


DocTuple = tuple[str, str | None, str, dict[str, Any]]


def _is_dry_run(config) -> bool:
    return bool(getattr(config, "dry_run", False))


def _should_skip_existing(config) -> bool:
    return bool(config.skip_existing) and not _is_dry_run(config)


def _is_timeout_exceeded(start_time: float, max_seconds: float) -> bool:
    return max_seconds > 0 and (time.perf_counter() - start_time) > max_seconds


def _resolve_document_type_policy(
    *,
    extension: str,
    key: str,
    text: str,
    extra_metadata: dict[str, Any],
) -> str:
    doc_type = extra_metadata.get("document_type", "unknown")

    if extension in ("csv", "xlsx", "xls") and doc_type in (
        "maintenance_request",
        "maintenance_job",
    ):
        return doc_type

    key_lower = key.lower()
    if is_fire_risk_assessment(key, text):
        return DocumentTypes.FIRE_RISK_ASSESSMENT
    if is_bms_document(key):
        return DocumentTypes.OPERATIONAL_DOC
    if "planon" in key_lower:
        return DocumentTypes.PLANON_DATA
    return doc_type


def _is_fra_candidate_policy(key: str, text: str) -> bool:
    return is_fire_risk_assessment(key, text)


class FraVectorExtractor(Protocol):
    def __call__(
        self,
        ctx: "IngestContext",
        *,
        base_path: str,
        key: str,
        text_sample: str,
        building: str,
        content_hash: str | None,
        file_id: str,
        processing_token: str,
        start_time: float,
        vectors_to_upsert: list[dict[str, Any]],
        parse_pool: ProcessPoolExecutor | None,
    ) -> FraVectorExtractResult: ...


def make_file_id(source_path: str, source_key: str, content_hash: str | None = None) -> str:
    """Build a stable ID for a source file."""
    if content_hash is not None and not isinstance(content_hash, str):
        raise ValidationError("content_hash must be a hex string or None")
    base = f"{source_path}:{source_key}"
    if content_hash:
        base += f":{content_hash}"
    return hashlib.sha256(base.encode("utf-8")).hexdigest()


def _doc_sub_id(doc_key: str) -> str:
    """Build a stable compact identifier for per-building/per-row doc keys."""
    return hashlib.sha1(doc_key.encode("utf-8")).hexdigest()[:12]


def make_id(file_id: str, doc_key: str, chunk_idx: int) -> str:
    """Generate a deterministic vector ID."""
    return f"{file_id}:{_doc_sub_id(doc_key)}:{chunk_idx}"


class DocumentProcessor:
    """Owns single-file processing and vector creation for ingestion."""

    def __init__(
        self,
        *,
        ctx: "IngestContext",
        base_path: str,
        alias_to_canonical: dict[str, str],
        fra_vector_extractor: FraVectorExtractor,
        building_resolver: "BuildingResolver",
    ) -> None:
        self.ctx = ctx
        self.base_path = base_path
        self.alias_to_canonical = alias_to_canonical
        self.fra_vector_extractor = fra_vector_extractor
        self.building_resolver = building_resolver
        self.cpu_pool: ProcessPoolExecutor | None = None
        self.parse_pool: ProcessPoolExecutor | None = None

    def set_cpu_pool(self, pool: ProcessPoolExecutor | None) -> None:
        self.cpu_pool = pool

    def set_parse_pool(self, pool: ProcessPoolExecutor | None) -> None:
        self.parse_pool = pool

    def load_bytes_and_ids(
        self,
        key: str,
    ) -> tuple[bytes | None, str | None, str | None, str | None]:
        return self._load_bytes_and_ids(key)

    def extract_text_and_chunks(self, key: str, data: bytes) -> tuple[str, list[str]]:
        return self._extract_text_and_chunks(key, data)

    def extract_docs_for_file(
        self,
        key: str,
        data: bytes,
        extension: str,
        *,
        text_sample: str | None = None,
    ) -> tuple[list[DocTuple], str, str | None, bool]:
        return self._extract_docs_for_file(
            key,
            data,
            extension,
            text_sample=text_sample,
        )

    def handle_dry_run(self, key: str, docs: list[DocTuple], file_id: str) -> bool:
        return self._handle_dry_run(key, docs, file_id)

    def maybe_extract_fra_vectors(
        self,
        *,
        key: str,
        text_sample: str,
        building: str | None,
        content_hash: str | None,
        file_id: str,
        processing_token: str,
        start_time: float,
        vectors_to_upsert: list[dict[str, Any]],
        is_fra_candidate: bool,
        docs: list[DocTuple],
    ) -> bool:
        return self._maybe_extract_fra_vectors(
            key=key,
            text_sample=text_sample,
            building=building,
            content_hash=content_hash,
            file_id=file_id,
            processing_token=processing_token,
            start_time=start_time,
            vectors_to_upsert=vectors_to_upsert,
            is_fra_candidate=is_fra_candidate,
            docs=docs,
        )

    def build_vectors_from_docs(
        self,
        *,
        key: str,
        extension: str,
        file_id: str,
        content_hash: str,
        processing_token: str,
        start_time: float,
        docs: list[DocTuple],
        precomputed_chunks: dict[str, list[str]] | None = None,
    ) -> list[dict[str, Any]]:
        return self._build_vectors_from_docs(
            key=key,
            extension=extension,
            file_id=file_id,
            content_hash=content_hash,
            processing_token=processing_token,
            start_time=start_time,
            docs=docs,
            precomputed_chunks=precomputed_chunks,
        )

    def _extract_text_and_chunks(self, key: str, data: bytes) -> tuple[str, list[str]]:
        start = time.perf_counter()
        if self.cpu_pool is not None:
            future = self.cpu_pool.submit(
                extract_text_and_chunk_in_process,
                key,
                data,
                embed_model=self.ctx.config.embed_model,
                chunk_tokens=self.ctx.config.chunk_tokens,
                chunk_overlap=self.ctx.config.chunk_overlap,
            )
            text_sample, chunks = future.result()
            elapsed = time.perf_counter() - start
            self.ctx.logger.debug(
                "Extract+chunk (process) %s: %.3fs (%d chunks)",
                key,
                elapsed,
                len(chunks),
            )
            return text_sample, chunks

        text_sample = extract_text(
            key,
            data,
            logger=self.ctx.logger,
        )
        chunks = chunk_text(self.ctx, text_sample)
        elapsed = time.perf_counter() - start
        self.ctx.logger.debug(
            "Extract+chunk (thread) %s: %.3fs (%d chunks)",
            key,
            elapsed,
            len(chunks),
        )
        return text_sample, chunks

    def skip_large_file(self, key: str, size_mb: float) -> bool:
        if self.ctx.config.max_file_mb > 0 and size_mb > self.ctx.config.max_file_mb:
            self.ctx.logger.info(
                "Skipping %s (%.2f MB > limit)", key, size_mb)
            return True
        return False

    def _load_bytes_and_ids(
        self,
        key: str,
    ) -> tuple[bytes | None, str | None, str | None, str | None]:
        data = fetch_bytes_secure(
            self.base_path,
            key,
            logger=self.ctx.logger,
        )
        content_hash = hashlib.sha256(data).hexdigest()
        file_id = make_file_id(self.base_path, key, content_hash)

        try:
            self.ctx.file_registry.record_discovered(
                file_id=file_id,
                source_path=self.base_path,
                source_key=key,
                content_hash=content_hash,
            )
        except (RedisError, OSError, ValueError, TypeError) as error:
            self.ctx.logger.warning(
                "FileRegistry discovery failed for %s: %s", key, error)

        if _should_skip_existing(self.ctx.config):
            try:
                if self.ctx.file_registry.is_success(file_id):
                    self.ctx.logger.debug(
                        "Skipping already indexed file (registry hit): %s", key)
                    return None, None, None, None
            except (RedisError, OSError, ValueError, TypeError) as error:
                self.ctx.logger.warning(
                    "FileRegistry check failed for %s: %s", key, error)

        processing_token = uuid.uuid4().hex
        try:
            started = self.ctx.file_registry.try_start_processing(
                file_id=file_id,
                lease_seconds=INGEST_PROCESSING_LEASE_SECONDS,
                processing_token=processing_token,
                source_path=self.base_path,
                source_key=key,
                content_hash=content_hash,
            )
            if not started:
                raise IngestError(
                    f"File already processing; aborting: {key}")
        except (RedisError, OSError, ValueError, TypeError) as error:
            self.ctx.logger.warning(
                "FileRegistry processing start failed for %s: %s", key, error)

        return data, content_hash, file_id, processing_token

    def _extract_docs_for_file(
        self,
        key: str,
        data: bytes,
        extension: str,
        *,
        text_sample: str | None = None,
    ) -> tuple[list[DocTuple], str, str | None, bool]:
        """
        Extract documents from a file.

        :param self: The instance of the class.
        :param key: The key of the file.
        :type key: str
        :param data: The file data.
        :type data: bytes
        :param extension: The file extension.
        :type extension: str
        :param text_sample: A sample of the text extracted from the file.
        :type text_sample: str | None
        :return: A tuple containing the extracted documents, the text sample, the building, and a flag indicating if it's a fire risk assessment candidate.
        :rtype: tuple[list[DocTuple], str, str | None, bool]
        """
        docs: list[DocTuple]
        text_sample = ""
        building: str | None = "Unknown"
        is_fra_candidate = False

        if extension in ("csv", "xlsx", "xls"):
            if "maintenance" in key.lower():
                docs = extract_maintenance_csv(
                    key,
                    data,
                    self.alias_to_canonical,
                    logger=self.ctx.logger,
                )
            else:
                docs = extract_text_csv_by_building_enhanced(
                    key,
                    data,
                    self.alias_to_canonical,
                    logger=self.ctx.logger,
                )
        else:
            if text_sample is None:
                text_sample = extract_text(
                    key,
                    data,
                    logger=self.ctx.logger,
                )
            resolution = self.building_resolver.resolve(key, text_sample)
            building = resolution.canonical
            confidence = resolution.confidence
            source = resolution.source

            flag_review = should_flag_for_review(confidence, source)
            building_metadata = {
                "canonical_building_name": building if building != "Unknown" else None,
                "building_confidence": confidence,
                "building_source": source,
                "building_flag_review": flag_review,
            }

            if confidence < INGEST_LOW_CONFIDENCE_WARN:
                self.ctx.logger.warning(
                    "Low confidence building assignment: %s -> %s (confidence: %.2f%%, source: %s)",
                    key,
                    building,
                    confidence * 100,
                    source,
                )

            docs = [(key, building, text_sample, building_metadata)]
            is_fra_candidate = _is_fra_candidate_policy(key, text_sample)
            if is_fra_candidate:
                fra_metadata = extract_fra_metadata(text_sample)
                building_metadata.update({
                    **fra_metadata,
                    "document_type": DocumentTypes.FIRE_RISK_ASSESSMENT,
                })

        return docs, text_sample, building, is_fra_candidate

    def _handle_dry_run(self, key: str, docs: list[DocTuple], file_id: str) -> bool:  # pylint: disable=unused-argument
        if not _is_dry_run(self.ctx.config):
            return False

        planned_vectors = 0
        extension = ext(key)

        for doc_key, canonical, text, extra_metadata in docs:
            # Resolve doc_type + namespace for prefix check
            doc_type = self._resolve_document_type(
                extension=extension,
                key=key,
                text=text,
                extra_metadata=extra_metadata,
            )
            resolved_namespace = self._route_namespace(
                doc_type)
            # No prefix checks: FileRegistry is the sole authority.

            doc_planned_vectors = 0
            first_chunk: str | None = None
            for chunk in chunk_text_generator(self.ctx, text):
                if first_chunk is None:
                    first_chunk = chunk
                doc_planned_vectors += 1
            planned_vectors += doc_planned_vectors

            if first_chunk is not None:
                metadata = {
                    "source_path": self.base_path,
                    "key": doc_key,
                    "source": key,
                    "document_type": extra_metadata.get("document_type", "unknown"),
                    "canonical_building_name": canonical,
                    **extra_metadata,
                    "text": first_chunk,
                }
                valid, reason = validate_with_truncation(
                    self.ctx,
                    metadata,
                    logger=self.ctx.logger,
                )
                if not valid:
                    self.ctx.logger.warning(
                        "Dry-run metadata invalid for %s: %s", key, reason)

        self.ctx.logger.info(
            "Dry-run planned vectors for %s: %d", key, planned_vectors
        )
        if planned_vectors > 0:
            self.ctx.stats.increment("total_vectors", planned_vectors)
        return True

    def _maybe_extract_fra_vectors(
        self,
        *,
        key: str,
        text_sample: str,
        building: str | None,
        content_hash: str | None,
        file_id: str,
        processing_token: str,
        start_time: float,
        vectors_to_upsert: list[dict[str, Any]],
        is_fra_candidate: bool,
        docs: list[DocTuple],
    ) -> bool:
        """If it's a fire risk assessment, extract FRA-specific vectors and skip normal ingestion."""
        if not (is_fra_candidate and building and building != "Unknown"):
            return False

        extracted_result = self.fra_vector_extractor(
            self.ctx,
            base_path=self.base_path,
            key=key,
            text_sample=text_sample,
            building=building,
            content_hash=content_hash,
            file_id=file_id,
            processing_token=processing_token,
            start_time=start_time,
            vectors_to_upsert=vectors_to_upsert,
            parse_pool=self.parse_pool,
        )
        extracted = extracted_result["added"]
        self.ctx.stats.increment("fra_risk_items_extracted", extracted)
        # ESCALATION / DIAGNOSTICS PATH: no risk items extracted
        if extracted == 0:
            try:
                _, _, _, extra_metadata = docs[0]

                # Always store parsing diagnostics
                extra_metadata["fra_parsing_confidence"] = extracted_result.get(
                    "parsing_confidence")
                extra_metadata["fra_parsing_warnings"] = extracted_result.get(
                    "parsing_warnings")
                extra_metadata["fra_parsing_field_scores"] = extracted_result.get(
                    "parsing_field_scores")

                # Set extraction status
                if extracted_result.get("missing_action_plan"):
                    extra_metadata["fra_action_plan_missing"] = True
                    extra_metadata["fra_risk_item_extraction_status"] = "no_action_plan_found"
                else:
                    extra_metadata["fra_risk_item_extraction_status"] = "no_risk_items_extracted"

            except (IndexError, KeyError, TypeError):
                self.ctx.logger.warning(
                    "Could not attach FRA extraction diagnostics for %s", key
                )

            # Optional escalation event (only for missing action plan)
            if (
                extracted_result.get("missing_action_plan")
                and getattr(self.ctx.config, "export_events", False)
                and self.ctx.config.export_events_file
            ):
                event = {
                    "event_type": "fra_action_plan_missing",
                    "file": key,
                    "canonical_building_name": building,
                    "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
                    "fra_parsing_confidence": extracted_result.get("parsing_confidence"),
                    "fra_parsing_warnings": extracted_result.get("parsing_warnings"),
                }
                with self.ctx.export_events_lock:
                    with open(self.ctx.config.export_events_file, "a", encoding="utf-8") as f:
                        f.write(json.dumps(event, ensure_ascii=False) + "\n")

            # IMPORTANT: fall back to normal document ingestion
            return False

        # Normal path: extracted risk items → keep current early-return behavior
        return True

    def _resolve_document_type(
        self,
        *,
        extension: str,
        key: str,
        text: str,
        extra_metadata: dict[str, Any],
    ) -> str:
        return _resolve_document_type_policy(
            extension=extension,
            key=key,
            text=text,
            extra_metadata=extra_metadata,
        )

    @staticmethod
    def _route_namespace(doc_type: str) -> str | None:
        return _route_namespace(doc_type)

    def _maybe_emit_building_event(
        self,
        *,
        key: str,
        canonical: str | None,
        doc_type: str,
        resolved_namespace: str,
    ) -> None:
        if not getattr(self.ctx.config, "export_events", False):
            return

        try:
            event = {
                "event_type": "building_assignment",
                "file": key,
                "canonical_building_name": canonical,
                "document_type": doc_type,
                "namespace_guess": resolved_namespace,
                "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
            }
            event_path = getattr(self.ctx.config, "export_events_file")
            if event_path:
                line = json.dumps(event, ensure_ascii=False) + "\n"
                with self.ctx.export_events_lock:
                    with open(event_path, "a", encoding="utf-8") as file_handle:
                        file_handle.write(line)
        except (OSError, ValueError, TypeError) as error:
            self.ctx.logger.warning(
                "Could not write building event for %s: %s", key, error)

    def _build_vectors_from_docs(
        self,
        *,
        key: str,
        extension: str,
        file_id: str,
        content_hash: str,
        processing_token: str,
        start_time: float,
        docs: list[DocTuple],
        precomputed_chunks: dict[str, list[str]] | None = None,
    ) -> list[dict[str, Any]]:
        vectors_to_upsert: list[dict[str, Any]] = []
        for doc_key, canonical, text, extra_metadata in docs:
            # Resolve doc_type + namespace *before* embedding
            doc_type = self._resolve_document_type(
                extension=extension,
                key=key,
                text=text,
                extra_metadata=extra_metadata,)
            resolved_namespace = self._route_namespace(doc_type)
            # Validate routing ONCE
            valid, reason = validate_namespace_routing(
                doc_type, resolved_namespace)
            if not valid:
                raise RoutingError(f"Invalid namespace routing: {reason}")

            batch_size = INGEST_EMBED_BATCH_SIZE
            batch_chunks: list[str] = []
            batch_indices: list[int] = []
            chunk_idx = 0

            def flush_batch(current_doc_key: str,
                            current_resolved_namespace: str | None,
                            current_canonical: str | None,
                            current_doc_type: str,
                            current_extra_metadata: dict[str, Any]) -> None:
                nonlocal batch_chunks, batch_indices
                if not batch_chunks:
                    return
                max_seconds = getattr(self.ctx.config, "max_file_seconds", 0)
                if _is_timeout_exceeded(start_time, max_seconds):
                    self.ctx.file_registry.mark_state(
                        file_id=file_id,
                        processing_token=processing_token,
                        status="failed",
                        error="file_timeout",
                        source_path=self.base_path,
                        source_key=key,
                        content_hash=content_hash,
                    )
                    raise IngestError(f"File processing timed out: {key}")
                valid_chunks: list[tuple[int, str, dict[str, Any]]] = []

                for index, chunk in zip(batch_indices, batch_chunks):
                    metadata = {
                        "source_path": self.base_path,
                        "key": current_doc_key,
                        "source": key,
                        "document_type": current_doc_type,
                        "canonical_building_name": current_canonical,
                        "content_hash": content_hash,
                        "text": chunk,
                    }
                    # Never trust namespace from extra_metadata
                    if "namespace" in current_extra_metadata:
                        current_extra_metadata = {
                            k: v for k, v in current_extra_metadata.items() if k != "namespace"
                        }
                    metadata.update(current_extra_metadata)
                    if current_canonical:
                        metadata = enrich_with_building_metadata(
                            metadata=metadata,
                            canonical=current_canonical,
                            ctx=self.ctx,
                            doc_type=current_doc_type,
                            chunk_idx=index,
                        )
                    sample_metadata = random.random() < 0.1
                    if sample_metadata:
                        try:
                            metadata_bytes = len(
                                json.dumps(
                                    metadata, ensure_ascii=False, default=str)
                            )
                            self.ctx.stats.observe_timing(
                                "metadata_bytes", metadata_bytes
                            )
                            metadata_no_text = {
                                k: v for k, v in metadata.items() if k != "text"
                            }
                            metadata_bytes_ex_text = len(
                                json.dumps(metadata_no_text,
                                           ensure_ascii=False, default=str)
                            )
                            self.ctx.stats.observe_timing(
                                "metadata_bytes_ex_text", metadata_bytes_ex_text
                            )
                        except (TypeError, ValueError):
                            pass

                    valid, reason = validate_with_truncation(
                        self.ctx,
                        metadata,
                        logger=self.ctx.logger,
                    )
                    if not valid:
                        self.ctx.logger.warning(
                            "Invalid metadata for %s: %s", key, reason)
                        self.ctx.stats.increment("vectors_invalid_metadata")
                        self.ctx.stats.append_failed(f"{key}:chunk_{index}")
                        continue
                    if sample_metadata:
                        try:
                            metadata_bytes_post_validation = len(
                                json.dumps(
                                    metadata, ensure_ascii=False, default=str)
                            )
                            self.ctx.stats.observe_timing(
                                "metadata_bytes_post_validation",
                                metadata_bytes_post_validation,
                            )
                        except (TypeError, ValueError):
                            pass

                    valid_chunks.append((index, chunk, metadata))

                if not valid_chunks:
                    batch_chunks = []
                    batch_indices = []
                    return

                embed_start = time.perf_counter()
                result = embed_texts_batch(
                    self.ctx,
                    [chunk for _, chunk, _ in valid_chunks],
                )
                embed_elapsed = time.perf_counter() - embed_start
                self.ctx.logger.debug(
                    "Embedding batch %s: %.3fs (%d chunks, %d valid)",
                    current_doc_key,
                    embed_elapsed,
                    len(batch_chunks),
                    len(valid_chunks),
                )
                if _is_timeout_exceeded(start_time, max_seconds):
                    self.ctx.file_registry.mark_state(
                        file_id=file_id,
                        processing_token=processing_token,
                        status="failed",
                        error="file_timeout",
                        source_path=self.base_path,
                        source_key=key,
                        content_hash=content_hash,
                    )
                    raise IngestError(f"File processing timed out: {key}")
                for idx_in_batch, (index, _chunk, metadata) in enumerate(valid_chunks):
                    if idx_in_batch in result.errors_by_index:
                        self.ctx.stats.increment("vectors_embedding_failed")
                        self.ctx.stats.append_failed(f"{key}:chunk_{index}")
                        continue
                    embedding = result.embeddings_by_index.get(idx_in_batch)
                    if embedding is None:
                        raise ExternalServiceError(
                            f"Embedding missing for index {idx_in_batch}"
                        )
                    vector_id = make_id(file_id, current_doc_key, index)

                    if (
                        getattr(self.ctx.config, "export_events", False)
                        and index == 0
                        and current_resolved_namespace
                    ):
                        self._maybe_emit_building_event(
                            key=key,
                            canonical=current_canonical,
                            doc_type=current_doc_type,
                            resolved_namespace=current_resolved_namespace,
                        )

                    self.ctx.logger.info(
                        "File %s -> doc_type=%s -> namespace=%s",
                        key,
                        current_doc_type,
                        current_resolved_namespace,
                    )

                    vectors_to_upsert.append(
                        {
                            "id": vector_id,
                            "values": embedding,
                            "metadata": metadata,
                            "namespace": current_resolved_namespace,
                            "_processing_token": processing_token,
                        }
                    )

                batch_chunks = []
                batch_indices = []

            chunks_override = (
                precomputed_chunks.get(doc_key) if precomputed_chunks else None
            )
            if chunks_override is not None:
                for chunk in chunks_override:
                    batch_chunks.append(chunk)
                    batch_indices.append(chunk_idx)
                    chunk_idx += 1
                    if len(batch_chunks) >= batch_size:
                        flush_batch(
                            doc_key,
                            resolved_namespace,
                            canonical,
                            doc_type,
                            extra_metadata,
                        )
            else:
                for chunk in chunk_text_generator(self.ctx, text):
                    batch_chunks.append(chunk)
                    batch_indices.append(chunk_idx)
                    chunk_idx += 1
                    if len(batch_chunks) >= batch_size:
                        flush_batch(
                            doc_key,
                            resolved_namespace,
                            canonical,
                            doc_type,
                            extra_metadata,
                        )

            flush_batch(doc_key, resolved_namespace,
                        canonical, doc_type, extra_metadata)

        return vectors_to_upsert


class FileCoordinator:
    """Registry + leasing + skip decisions for a single file."""

    def __init__(self, processor: DocumentProcessor) -> None:
        self._processor = processor

    def prepare(
        self,
        obj: dict[str, Any],
    ) -> tuple[str, str, bytes, str, str, str] | None:
        key = obj["Key"]
        size_mb = obj.get("Size", 0) / (1024 * 1024)
        extension = ext(key)
        if self._processor.skip_large_file(key, size_mb):
            return None

        data, content_hash, file_id, processing_token = (
            self._processor.load_bytes_and_ids(key)
        )
        if (
            data is None
            or content_hash is None
            or file_id is None
            or processing_token is None
        ):
            return None

        return key, extension, data, content_hash, file_id, processing_token


class Extractor:
    """Bytes -> text/pages -> doc tuples."""

    def __init__(self, processor: DocumentProcessor) -> None:
        self._processor = processor

    def extract(
        self,
        *,
        key: str,
        extension: str,
        data: bytes,
    ) -> tuple[list[DocTuple], str | None, str | None, bool, dict[str, list[str]] | None]:
        precomputed_chunks: dict[str, list[str]] | None = None
        if extension not in ("csv", "xlsx", "xls"):
            text_sample, chunks = self._processor.extract_text_and_chunks(
                key, data)
            precomputed_chunks = {key: chunks}
        else:
            text_sample = None

        docs, text_sample, building, is_fra_candidate = (
            self._processor.extract_docs_for_file(
                key,
                data,
                extension,
                text_sample=text_sample,
            )
        )

        return docs, text_sample, building, is_fra_candidate, precomputed_chunks


class Vectoriser:
    """Documents -> chunks -> embeddings -> vectors."""

    def __init__(self, processor: DocumentProcessor) -> None:
        self._processor = processor

    def vectorise(
        self,
        *,
        key: str,
        extension: str,
        file_id: str,
        content_hash: str,
        processing_token: str,
        start_time: float,
        docs: list[DocTuple],
        text_sample: str | None,
        building: str | None,
        is_fra_candidate: bool,
        precomputed_chunks: dict[str, list[str]] | None,
    ) -> list[dict[str, Any]]:
        if self._processor.handle_dry_run(key, docs, file_id):
            return []

        vectors_to_upsert: list[dict[str, Any]] = []
        if self._processor.maybe_extract_fra_vectors(
            key=key,
            text_sample=text_sample or "",
            building=building,
            content_hash=content_hash,
            file_id=file_id,
            processing_token=processing_token,
            start_time=start_time,
            vectors_to_upsert=vectors_to_upsert,
            is_fra_candidate=is_fra_candidate,
            docs=docs,
        ):
            return vectors_to_upsert

        return self._processor.build_vectors_from_docs(
            key=key,
            extension=extension,
            file_id=file_id,
            content_hash=content_hash,
            processing_token=processing_token,
            start_time=start_time,
            docs=docs,
            precomputed_chunks=precomputed_chunks,
        )


class Writer:
    """Upsert + verify + state recording wrapper."""

    def __init__(self, ctx: "IngestContext", upsert_func) -> None:
        self._ctx = ctx
        self._upsert = upsert_func

    def write_batch(self, batch: list[dict[str, Any]]) -> None:
        self._upsert(self._ctx, batch)


class FileIngestOrchestrator:
    """Coordinates file ingestion using dedicated components."""

    def __init__(self, processor: DocumentProcessor) -> None:
        self._processor = processor
        self._coordinator = FileCoordinator(processor)
        self._extractor = Extractor(processor)
        self._vectoriser = Vectoriser(processor)

    def set_cpu_pool(self, pool: ProcessPoolExecutor | None) -> None:
        self._processor.set_cpu_pool(pool)

    def set_parse_pool(self, pool: ProcessPoolExecutor | None) -> None:
        self._processor.set_parse_pool(pool)

    def process(self, obj: dict[str, Any]) -> "FileProcessResult":
        start = time.perf_counter()
        prepared = self._coordinator.prepare(obj)
        if prepared is None:
            return FileProcessResult(status="skipped", vectors=[], vector_count=0)

        key, extension, data, content_hash, file_id, processing_token = prepared
        max_seconds = getattr(self._processor.ctx.config,
                              "max_file_seconds", 0)

        docs, text_sample, building, is_fra_candidate, precomputed_chunks = (
            self._extractor.extract(
                key=key,
                extension=extension,
                data=data,
            )
        )
        if _is_timeout_exceeded(start, max_seconds):
            self._processor.ctx.file_registry.mark_state(
                file_id=file_id,
                processing_token=processing_token,
                status="failed",
                error="file_timeout",
                source_path=self._processor.base_path,
                source_key=key,
                content_hash=content_hash,
            )
            raise IngestError(f"File processing timed out: {key}")

        vectors = self._vectoriser.vectorise(
            key=key,
            extension=extension,
            file_id=file_id,
            content_hash=content_hash,
            processing_token=processing_token,
            start_time=start,
            docs=docs,
            text_sample=text_sample,
            building=building,
            is_fra_candidate=is_fra_candidate,
            precomputed_chunks=precomputed_chunks,
        )
        if _is_timeout_exceeded(start, max_seconds):
            self._processor.ctx.file_registry.mark_state(
                file_id=file_id,
                processing_token=processing_token,
                status="failed",
                error="file_timeout",
                source_path=self._processor.base_path,
                source_key=key,
                content_hash=content_hash,
            )
            raise IngestError(f"File processing timed out: {key}")

        return FileProcessResult(
            status="processed",
            vectors=vectors,
            vector_count=len(vectors),
        )


@dataclass(frozen=True)
class FileProcessResult:
    status: str
    vectors: list[dict[str, Any]]
    vector_count: int
