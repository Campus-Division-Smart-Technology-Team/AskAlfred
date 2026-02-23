#!/usr/bin/env python3
"""
Convert Microsoft Word files (.doc, .docx) to PDF and move originals to Recycle Bin.
Windows-specific script using win32com for Word conversion and send2trash for safe deletion.
"""

import sys
import io
import logging
from pathlib import Path
from typing import List, Tuple
import argparse

# Fix Unicode encoding for Windows console to support emojis
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(
        sys.stderr.buffer, encoding='utf-8', errors='replace')

# Windows-specific imports
try:
    import win32com.client
    from pythoncom import CoInitialize, CoUninitialize
except ImportError:
    print("ERROR: win32com not found. Install with: pip install pywin32")
    sys.exit(1)

try:
    from send2trash import send2trash
except ImportError:
    print("ERROR: send2trash not found. Install with: pip install send2trash")
    sys.exit(1)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("word_to_pdf_conversion.log", encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

# Constants
DEFAULT_PATH = r"C:\Users\rd23091\Downloads\Alfred"
WORD_EXTENSIONS = {'.doc', '.docx'}
PDF_FORMAT = 17  # Word SaveAs PDF format constant

# Statistics tracking
stats = {
    'total_found': 0,
    'converted': 0,
    'failed': 0,
    'moved_to_recycle': 0,
    'failed_files': []
}


def find_word_documents(base_path: str, recursive: bool = True) -> List[Path]:
    """
    Find all Word documents in the specified directory.

    Args:
        base_path: Root directory to search
        recursive: If True, search subdirectories

    Returns:
        List of Path objects for Word documents
    """
    base_path_obj = Path(base_path)

    if not base_path_obj.exists():
        raise FileNotFoundError(f"Directory not found: {base_path}")

    if not base_path_obj.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {base_path}")

    word_files = []

    if recursive:
        # Search recursively
        for ext in WORD_EXTENSIONS:
            word_files.extend(base_path_obj.rglob(f'*{ext}'))
    else:
        # Search only in the root directory
        for ext in WORD_EXTENSIONS:
            word_files.extend(base_path_obj.glob(f'*{ext}'))

    # Filter out temporary Word files (starting with ~$)
    word_files = [f for f in word_files if not f.name.startswith('~$')]

    return sorted(word_files)


def convert_word_to_pdf(word_path: Path) -> Tuple[bool, str, Path]:
    """
    Convert a single Word document to PDF using Microsoft Word COM interface.

    Args:
        word_path: Path to the Word document

    Returns:
        Tuple of (success: bool, message: str, pdf_path: Path)
    """
    pdf_path = word_path.with_suffix('.pdf')

    # Check if PDF already exists
    if pdf_path.exists():
        logging.warning("PDF already exists: %s", pdf_path.name)
        return True, "PDF already exists (skipped conversion)", pdf_path

    # Initialize COM
    CoInitialize()
    word = None
    doc = None

    try:
        # Create Word application instance
        word = win32com.client.Dispatch("Word.Application")
        word.Visible = False  # Run in background
        word.DisplayAlerts = 0  # Disable alerts

        # Open the document
        doc = word.Documents.Open(str(word_path.absolute()))

        # Save as PDF
        doc.SaveAs(str(pdf_path.absolute()), FileFormat=PDF_FORMAT)

        logging.info("✅ Converted: %s -> %s", word_path.name, pdf_path.name)
        return True, "Conversion successful", pdf_path

    except Exception as e:  # pylint: disable=broad-except
        error_msg = f"Conversion failed: {str(e)}"
        logging.error("❌ Error converting %s: %s", word_path.name, error_msg)
        return False, error_msg, pdf_path

    finally:
        # Clean up
        try:
            if doc:
                doc.Close(SaveChanges=False)
            if word:
                word.Quit()
        except Exception as e:  # pylint: disable=broad-except
            logging.warning("Error closing Word: %s", e)

        # Uninitialize COM
        CoUninitialize()


def move_to_recycle_bin(file_path: Path) -> Tuple[bool, str]:
    """
    Move a file to the Windows Recycle Bin.

    Args:
        file_path: Path to the file to move

    Returns:
        Tuple of (success: bool, message: str)
    """
    try:
        send2trash(str(file_path))
        logging.info("🗑️  Moved to Recycle Bin: %s", file_path.name)
        return True, "Moved to Recycle Bin"

    except Exception as e:  # pylint: disable=broad-except
        error_msg = f"Failed to move to Recycle Bin: {str(e)}"
        logging.error("❌ %s - File: %s", error_msg, file_path.name)
        return False, error_msg


def process_word_documents(
    base_path: str,
    recursive: bool = True,
    keep_originals: bool = False,
    dry_run: bool = False
) -> None:
    """
    Main function to process all Word documents: convert to PDF and move to Recycle Bin.

    Args:
        base_path: Root directory containing Word documents
        recursive: If True, process subdirectories
        keep_originals: If True, don't move originals to Recycle Bin
        dry_run: If True, show what would be done without actually doing it
    """
    logging.info("=" * 70)
    logging.info("Word to PDF Converter")
    logging.info("=" * 70)
    logging.info("Base path: %s", base_path)
    logging.info("Recursive: %s", recursive)
    logging.info("Keep originals: %s", keep_originals)
    logging.info("Dry run: %s", dry_run)
    logging.info("=" * 70)

    # Find all Word documents
    logging.info("Searching for Word documents...")
    try:
        word_files = find_word_documents(base_path, recursive)
    except (FileNotFoundError, NotADirectoryError) as e:
        logging.error(str(e))
        return

    stats['total_found'] = len(word_files)

    if not word_files:
        logging.info("No Word documents found.")
        return

    logging.info("Found %d Word document(s)", len(word_files))

    if dry_run:
        logging.info("\n🔍 DRY RUN MODE - No files will be modified\n")

    # Process each file
    for i, word_file in enumerate(word_files, 1):
        logging.info("\n[%d/%d] Processing: %s", i,
                     len(word_files), word_file.name)
        logging.info("    Location: %s", word_file.parent)

        if dry_run:
            logging.info("    Would convert to: %s",
                         word_file.with_suffix('.pdf').name)
            if not keep_originals:
                logging.info("    Would move to Recycle Bin: %s",
                             word_file.name)
            continue

        # Convert to PDF
        success, _, _ = convert_word_to_pdf(word_file)

        if success:
            stats['converted'] += 1

            # Move original to Recycle Bin if requested
            if not keep_originals:
                move_success, move_message = move_to_recycle_bin(word_file)
                if move_success:
                    stats['moved_to_recycle'] += 1
                else:
                    logging.warning("    ⚠️  %s", move_message)
        else:
            stats['failed'] += 1
            stats['failed_files'].append(str(word_file))

    # Print summary
    print_summary()


def print_summary() -> None:
    """Print processing summary statistics."""
    logging.info("\n%s", "=" * 70)
    logging.info("CONVERSION SUMMARY")
    logging.info("=" * 70)
    logging.info("Word documents found:       %d", stats['total_found'])
    logging.info("Successfully converted:     %d", stats['converted'])
    logging.info("Failed conversions:         %d", stats['failed'])
    logging.info("Moved to Recycle Bin:       %d", stats['moved_to_recycle'])
    logging.info("=" * 70)

    if stats['failed_files']:
        logging.info("\nFailed files:")
        for failed_file in stats['failed_files']:
            logging.info("  ❌ %s", failed_file)

    if stats['converted'] > 0:
        logging.info("\n✅ Conversion complete!")
    elif stats['total_found'] == 0:
        logging.info("\n📭 No Word documents found.")
    else:
        logging.info("\n⚠️  No files were converted.")


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert Word documents to PDF and move originals to Recycle Bin",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert all Word documents in default path recursively
  python word_to_pdf.py
  
  # Convert documents in specific directory
  python word_to_pdf.py --path "C:\\Documents\\MyFolder"
  
  # Convert only in root directory (not subdirectories)
  python word_to_pdf.py --no-recursive
  
  # Keep original Word files (don't move to Recycle Bin)
  python word_to_pdf.py --keep-originals
  
  # Dry run (show what would be done without doing it)
  python word_to_pdf.py --dry-run
        """
    )

    parser.add_argument(
        '--path',
        type=str,
        default=DEFAULT_PATH,
        help=f'Path to directory containing Word documents (default: {DEFAULT_PATH})'
    )

    parser.add_argument(
        '--no-recursive',
        action='store_true',
        help='Do not search subdirectories'
    )

    parser.add_argument(
        '--keep-originals',
        action='store_true',
        help='Keep original Word files (do not move to Recycle Bin)'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without actually doing it'
    )

    return parser.parse_args()


def check_requirements() -> bool:
    """
    Check if Microsoft Word is installed and accessible.

    Returns:
        True if Word is available, False otherwise
    """
    try:
        CoInitialize()
        word = win32com.client.Dispatch("Word.Application")
        word.Quit()
        CoUninitialize()
        return True
    except Exception as e:  # pylint: disable=broad-except
        logging.error("Microsoft Word is not available: %s", e)
        logging.error(
            "Please ensure Microsoft Word is installed on this system.")
        return False


def main():
    """Main entry point."""
    # Check requirements
    if not check_requirements():
        sys.exit(1)

    # Parse arguments
    args = parse_arguments()

    # Process documents
    try:
        process_word_documents(
            base_path=args.path,
            recursive=not args.no_recursive,
            keep_originals=args.keep_originals,
            dry_run=args.dry_run
        )
    except KeyboardInterrupt:
        logging.info("\n\n⚠️  Operation cancelled by user")
        sys.exit(0)
    except Exception as e:  # pylint: disable=broad-except
        logging.error("\n❌ Unexpected error: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
