#!/usr/bin/env python3
"""
Enhanced Index to CSV Exporter
Pulls ALL metadata from Pinecone index including all namespaces and fields
"""

import os
import csv
import json
import logging
import argparse
from typing import Optional, List, Dict, Any, Set
from datetime import datetime
from dotenv import load_dotenv
from pinecone import Pinecone

# ---------------- Env & constants ----------------
load_dotenv()

# ---- logging ----
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(message)s",
)

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
if not PINECONE_API_KEY:
    raise RuntimeError("PINECONE_API_KEY is not set")

# Initialise Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)


def get_all_namespaces(index) -> List[str]:
    """
    Get all namespaces in the index.

    Returns:
        List of namespace names (empty string for default namespace)
    """
    try:
        stats = index.describe_index_stats()
        namespaces = []

        if hasattr(stats, 'namespaces') and stats.namespaces:
            for ns_name, ns_stats in stats.namespaces.items():
                if ns_stats.vector_count > 0:
                    namespaces.append(ns_name if ns_name else "")

        # Always include default namespace (empty string)
        if "" not in namespaces:
            namespaces.append("")

        logging.info("Found %d namespaces: %s", len(namespaces),
                     [ns if ns else "default" for ns in namespaces])
        return namespaces

    except Exception as e:
        logging.warning("Error getting namespaces: %s. Using default only.", e)
        return [""]


def get_all_vector_ids(index, namespace: Optional[str] = None) -> List[str]:
    """Get all vector IDs from the index using the list method."""
    all_ids = []

    try:
        # Use list() method to iterate through all vector IDs
        for ids_batch in index.list(namespace=namespace):
            if ids_batch and len(ids_batch) > 0:
                all_ids.extend(ids_batch)
            else:
                break  # No more IDs to fetch

        ns_display = namespace if namespace else "default"
        logging.info("Found %d vector IDs in namespace '%s'",
                     len(all_ids), ns_display)
        return all_ids

    except Exception as e:
        logging.error("Error listing vector IDs: %s", e)
        return []


def discover_all_metadata_fields(
    index,
    all_ids: List[str],
    namespace: Optional[str] = None,
    sample_size: int = 500
) -> Set[str]:
    """
    Discover ALL metadata fields by sampling vectors.

    Args:
        index: Pinecone index object
        all_ids: List of all vector IDs
        namespace: Namespace to query
        sample_size: Number of vectors to sample for field discovery

    Returns:
        Set of all unique metadata field names
    """
    all_fields = set()

    # Sample from beginning, middle, and end to capture all field variations
    sample_indices = []
    if len(all_ids) <= sample_size:
        sample_indices = list(range(len(all_ids)))
    else:
        # Sample from start
        sample_indices.extend(range(min(sample_size // 3, len(all_ids))))
        # Sample from middle
        mid_start = len(all_ids) // 2 - sample_size // 6
        mid_end = len(all_ids) // 2 + sample_size // 6
        sample_indices.extend(
            range(max(0, mid_start), min(len(all_ids), mid_end)))
        # Sample from end
        sample_indices.extend(
            range(max(0, len(all_ids) - sample_size // 3), len(all_ids)))

    sample_ids = [all_ids[i] for i in sample_indices]

    logging.info(
        "Sampling %d vectors to discover metadata fields...", len(sample_ids))

    # Fetch in batches
    batch_size = 100
    for i in range(0, len(sample_ids), batch_size):
        batch = sample_ids[i:i + batch_size]
        try:
            response = index.fetch(ids=batch, namespace=namespace)
            if response and response.vectors:
                for vector_data in response.vectors.values():
                    metadata = vector_data.metadata or {}
                    all_fields.update(metadata.keys())
        except Exception as e:
            logging.warning("Error fetching sample batch: %s", e)
            continue

    logging.info("Discovered %d unique metadata fields", len(all_fields))
    return all_fields


def format_value_for_csv(value: Any) -> str:
    """
    Format a metadata value for CSV output.

    Args:
        value: Any metadata value

    Returns:
        String representation suitable for CSV
    """
    if value is None or value == "":
        return ""

    # Handle lists - convert to JSON string for readability
    if isinstance(value, list):
        # Check if it's a simple list of strings/numbers
        if all(isinstance(item, (str, int, float, bool)) for item in value):
            return json.dumps(value, ensure_ascii=False)
        else:
            return json.dumps(value, ensure_ascii=False, default=str)

    # Handle dicts - convert to JSON string
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, default=str)

    # Handle booleans
    if isinstance(value, bool):
        return str(value)

    # Handle numbers
    if isinstance(value, (int, float)):
        return str(value)

    # Everything else to string
    return str(value)


def export_namespace_to_csv(
    index,
    namespace: Optional[str],
    all_ids: List[str],
    metadata_fields: Set[str],
    output_path: str,
    append_mode: bool = False
) -> int:
    """
    Export a single namespace to CSV.

    Args:
        index: Pinecone index object
        namespace: Namespace to export (None for default)
        all_ids: List of vector IDs in this namespace
        metadata_fields: Set of all metadata field names
        output_path: Path to output CSV file
        append_mode: Whether to append to existing file

    Returns:
        Number of vectors processed
    """
    ns_display = namespace if namespace else "default"

    if not all_ids:
        logging.warning("No vectors to export for namespace '%s'", ns_display)
        return 0

    # Sort fields for consistent ordering
    # Important fields first, then alphabetically
    priority_fields = [
        "document_type",
        "canonical_building_name",
        "building_name",
        "building_aliases",
        "key",
        "source",
        "text"
    ]

    # Start with priority fields that exist, then add rest alphabetically
    header_fields = ["ID", "namespace"]
    for field in priority_fields:
        if field in metadata_fields:
            header_fields.append(field)

    # Add remaining fields alphabetically
    remaining_fields = sorted(metadata_fields - set(priority_fields))
    header_fields.extend(remaining_fields)

    logging.info("Exporting namespace '%s' with %d metadata fields",
                 ns_display, len(metadata_fields))

    # Open file in appropriate mode
    mode = 'a' if append_mode else 'w'

    with open(output_path, mode=mode, newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)

        # Write header (only if not appending or file is empty)
        if not append_mode:
            csv_writer.writerow(header_fields)

        # Process vectors in batches
        batch_size = 100
        total_processed = 0
        total_batches = (len(all_ids) + batch_size - 1) // batch_size

        for i in range(0, len(all_ids), batch_size):
            batch_ids = all_ids[i:i + batch_size]
            batch_num = i // batch_size + 1
            start_id = i + 1
            end_id = min(i + batch_size, len(all_ids))

            try:
                logging.info(
                    "[%s] Batch %d/%d: Fetching vectors %d-%d (%d/%d completed)",
                    ns_display, batch_num, total_batches,
                    start_id, end_id, total_processed, len(all_ids)
                )

                response = index.fetch(ids=batch_ids, namespace=namespace)

                if not response or not response.vectors:
                    logging.warning(
                        "[%s] No vectors returned for batch %d",
                        ns_display, batch_num
                    )
                    continue

                # Process each vector in the batch
                batch_count = 0
                for vector_id, vector_data in response.vectors.items():
                    metadata = vector_data.metadata or {}

                    # Build row dynamically
                    # ID and namespace first
                    row_data = [vector_id, ns_display]

                    # Add all metadata fields in the same order as header
                    for field in header_fields[2:]:  # Skip ID and namespace
                        value = metadata.get(field, "")
                        formatted_value = format_value_for_csv(value)
                        row_data.append(formatted_value)

                    # Write the row
                    csv_writer.writerow(row_data)
                    total_processed += 1
                    batch_count += 1

                logging.info(
                    "[%s] Batch %d/%d completed: Wrote %d vectors",
                    ns_display, batch_num, total_batches, batch_count
                )

            except Exception as e:
                logging.error(
                    "[%s] Error processing batch %d: %s",
                    ns_display, batch_num, e
                )
                continue

    return total_processed


def export_index_to_csv(
    idx_name: str,
    output_path: str,
    namespace: Optional[str] = None,
    all_namespaces: bool = False,
    sample_size: int = 500
) -> None:
    """
    Export Pinecone index to CSV with ALL metadata fields.

    Args:
        idx_name: Name of the Pinecone index
        output_path: Path to save the CSV file
        namespace: Specific namespace to export (None = default)
        all_namespaces: If True, export all namespaces to single CSV
        sample_size: Number of vectors to sample for field discovery
    """
    try:
        index = pc.Index(idx_name)

        # Get index statistics
        stats = index.describe_index_stats()
        total_vectors = stats.total_vector_count if stats else 0

        logging.info("=" * 70)
        logging.info("PINECONE INDEX EXPORT")
        logging.info("=" * 70)
        logging.info("Index: %s", idx_name)
        logging.info("Total vectors: %d", total_vectors)
        logging.info("Output file: %s", output_path)
        logging.info("=" * 70)

        # Determine which namespaces to export
        if all_namespaces:
            namespaces_to_export = get_all_namespaces(index)
            logging.info("Exporting ALL namespaces: %s",
                         [ns if ns else "default" for ns in namespaces_to_export])
        else:
            namespaces_to_export = [namespace]
            ns_display = namespace if namespace else "default"
            logging.info("Exporting single namespace: %s", ns_display)

        # Collect all metadata fields across all namespaces
        all_metadata_fields = set()
        namespace_data = {}  # namespace -> list of IDs

        for ns in namespaces_to_export:
            ns_display = ns if ns else "default"
            logging.info("\n" + "=" * 70)
            logging.info("Processing namespace: %s", ns_display)
            logging.info("=" * 70)

            # Get all vector IDs for this namespace
            vector_ids = get_all_vector_ids(index, ns)

            if not vector_ids:
                logging.warning(
                    "No vectors found in namespace '%s', skipping", ns_display)
                continue

            namespace_data[ns] = vector_ids

            # Discover metadata fields from this namespace
            ns_fields = discover_all_metadata_fields(
                index, vector_ids, ns, sample_size
            )
            all_metadata_fields.update(ns_fields)

            logging.info("Namespace '%s': %d vectors, %d metadata fields",
                         ns_display, len(vector_ids), len(ns_fields))

        if not namespace_data:
            logging.error("No data to export!")
            return

        logging.info("\n" + "=" * 70)
        logging.info("METADATA FIELD SUMMARY")
        logging.info("=" * 70)
        logging.info("Total unique metadata fields across all namespaces: %d",
                     len(all_metadata_fields))
        logging.info("\nAll fields:")
        for field in sorted(all_metadata_fields):
            logging.info("  - %s", field)
        logging.info("=" * 70)

        # Export each namespace
        total_exported = 0
        append_mode = False

        for ns, vector_ids in namespace_data.items():
            ns_display = ns if ns else "default"
            logging.info("\n" + "=" * 70)
            logging.info("EXPORTING NAMESPACE: %s", ns_display)
            logging.info("=" * 70)

            count = export_namespace_to_csv(
                index=index,
                namespace=ns,
                all_ids=vector_ids,
                metadata_fields=all_metadata_fields,
                output_path=output_path,
                append_mode=append_mode
            )

            total_exported += count
            append_mode = True  # Subsequent namespaces append to file

            logging.info("Namespace '%s' complete: %d vectors exported",
                         ns_display, count)

        # Final summary
        logging.info("\n" + "=" * 70)
        logging.info("EXPORT COMPLETED SUCCESSFULLY!")
        logging.info("=" * 70)
        logging.info("Total vectors exported: %d", total_exported)
        logging.info("Total namespaces: %d", len(namespace_data))
        logging.info("Total metadata fields: %d", len(all_metadata_fields))
        logging.info("Output file: %s", output_path)
        logging.info("File size: %.2f MB", os.path.getsize(
            output_path) / (1024 * 1024))
        logging.info("=" * 70)

    except Exception as e:
        logging.error("Error exporting index '%s': %s", idx_name, e)
        raise


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Export Pinecone index to CSV with all metadata fields"
    )
    parser.add_argument(
        "--index",
        default="local-docs",
        help="Pinecone index name (default: local-docs)"
    )
    parser.add_argument(
        "--output",
        default="pinecone_export.csv",
        help="Output CSV file path (default: pinecone_export.csv)"
    )
    parser.add_argument(
        "--namespace",
        default=None,
        help="Specific namespace to export (default: all namespaces)"
    )
    parser.add_argument(
        "--all-namespaces",
        action="store_true",
        help="Export all namespaces to a single CSV file"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=500,
        help="Number of vectors to sample for field discovery (default: 500)"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    try:
        # Add timestamp to output filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = args.output.rsplit('.', 1)[0]
        ext = args.output.rsplit('.', 1)[1] if '.' in args.output else 'csv'
        output_file = f"{base_name}_{timestamp}.{ext}"

        export_index_to_csv(
            idx_name=args.index,
            output_path=output_file,
            namespace=args.namespace,
            all_namespaces=args.all_namespaces or args.namespace is None,
            sample_size=args.sample_size
        )

        print(f"\n✅ Export completed successfully!")
        print(f"📄 Output file: {output_file}")

    except KeyboardInterrupt:
        logging.warning("\n⚠️  Export interrupted by user")
        print("\n⚠️  Export interrupted")
    except Exception as e:
        logging.error("Export failed: %s", e, exc_info=True)
        print(f"\n❌ Export failed: {e}")
        exit(1)
