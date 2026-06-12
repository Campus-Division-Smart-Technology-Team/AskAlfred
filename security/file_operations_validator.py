#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File Operations Security Validator for Alfred.

Comprehensive security checks for file operations including:
- Path traversal prevention
- Symlink attack prevention
- File size limits
- File type whitelisting
- Directory access control
- Rate limiting per directory

Implements OWASP secure file upload/download best practices.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Optional

# ===========================================================================
# CONSTANTS
# ===========================================================================

# Allowed file extensions for different operation types
ALLOWED_DOCUMENT_EXTENSIONS = {
    "pdf",
    "txt",
    "md",
    "docx",
    "doc",
    "xlsx",
    "xls",
    "pptx",
    "ppt",
    "json",
    "csv",
}

ALLOWED_INGEST_EXTENSIONS = {
    "pdf",
    "txt",
    "docx",
    "doc",
    "xlsx",
    "xls",
    "pptx",
    "ppt",
    "json",
    "csv",
}

DANGEROUS_EXTENSIONS = {
    "exe",
    "bat",
    "cmd",
    "com",
    "pif",
    "scr",
    "vbs",
    "js",
    "jar",
    "zip",
    "rar",
    "7z",
    "sh",
    "bash",
    "py",
    "pyc",
    "so",
    "dll",
    "sys",
    "ini",
    "cfg",
    "conf",
}

# File size limits (in MB)
MAX_FILE_SIZE_MB = 100.0
MAX_BATCH_TOTAL_SIZE_MB = 5000.0

# Directory traversal patterns
SUSPICIOUS_PATTERNS = ["..", "~", "$", ";", "|", "&", "`", "$(", "${"]

# Rate limiting
FILE_OPERATION_RATE_LIMIT = 1000  # operations per minute per directory
FILE_OPERATION_WINDOW_SECONDS = 60

# ===========================================================================
# LOGGING
# ===========================================================================
logger = logging.getLogger(__name__)


# ===========================================================================
# EXCEPTIONS
# ===========================================================================


class FileOperationSecurityError(Exception):
    """Base exception for file operation security violations."""

    pass


class PathTraversalError(FileOperationSecurityError):
    """Raised when path traversal attack is detected."""

    pass


class SymlinkError(FileOperationSecurityError):
    """Raised when symlink is encountered."""

    pass


class FileSizeError(FileOperationSecurityError):
    """Raised when file exceeds size limits."""

    pass


class FileTypeError(FileOperationSecurityError):
    """Raised when file type is not allowed."""

    pass


class DirectoryAccessError(FileOperationSecurityError):
    """Raised when directory access is denied."""

    pass


# ===========================================================================
# CORE VALIDATION FUNCTIONS
# ===========================================================================


def is_safe_extension(
    filename: str,
    allowed_extensions: Optional[set[str]] = None,
) -> bool:
    """
    Check if file extension is safe.

    Args:
        filename: Filename to check
        allowed_extensions: set of allowed extensions (default: ALLOWED_DOCUMENT_EXTENSIONS)

    Returns:
        True if extension is allowed and not dangerous
    """
    if allowed_extensions is None:
        allowed_extensions = ALLOWED_DOCUMENT_EXTENSIONS

    if "." not in filename:
        return False

    ext = filename.rsplit(".", 1)[1].lower()

    # Check if dangerous
    if ext in DANGEROUS_EXTENSIONS:
        logger.warning("Dangerous file extension detected: %s", ext)
        return False

    # Check if in whitelist
    return ext in allowed_extensions


def has_suspicious_patterns(path_str: str) -> bool:
    """
    Check if path contains suspicious patterns.

    Args:
        path_str: Path string to check

    Returns:
        True if suspicious patterns detected
    """
    for pattern in SUSPICIOUS_PATTERNS:
        if pattern == "&":
            if (
                path_str.startswith("&")
                or path_str.endswith("&")
                or "&&" in path_str
                or " &" in path_str
                or "& " in path_str
            ):
                logger.warning("Suspicious pattern '%s' found in path", pattern)
                return True
            continue
        if pattern in path_str:
            logger.warning("Suspicious pattern '%s' found in path", pattern)
            return True
    return False


def validate_path_safety(
    base_path: str,
    relative_path: str,
    allow_directories: bool = False,
) -> Path:
    """
    Validate that a path is safe and within the base directory.

    Prevents:
    - Path traversal attacks (../)
    - Symlink following
    - Access outside base directory
    - Invalid file types

    Args:
        base_path: Base directory path
        relative_path: Relative path to validate
        allow_directories: Whether to allow directories or only files

    Returns:
        Resolved Path object

    Raises:
        PathTraversalError: If path tries to escape base directory
        SymlinkError: If path is a symlink
        DirectoryAccessError: If base directory is invalid
        FileOperationSecurityError: For other security violations
    """
    # Validate base path
    base = Path(base_path).resolve()

    if not base.exists():
        raise DirectoryAccessError(f"Base directory not found: {base_path}")

    if not base.is_dir():
        raise DirectoryAccessError(f"Base path is not a directory: {base_path}")

    # Check for suspicious patterns in relative path
    if has_suspicious_patterns(relative_path):
        raise PathTraversalError(f"Suspicious patterns in path: {relative_path}")

    # Resolve the target path
    target = (base / relative_path).resolve()

    # Verify target is within base directory
    try:
        target.relative_to(base)
    except ValueError as e:
        logger.warning("Path traversal detected: %s -> %s", relative_path, target)
        raise PathTraversalError(f"Path escapes base directory: {relative_path}") from e

    # Check for symlinks
    if target.is_symlink():
        raise SymlinkError(f"Symlinks not allowed: {relative_path}")

    # Check if path exists
    if not target.exists():
        raise FileOperationSecurityError(f"Path does not exist: {relative_path}")

    # Check type: directory or file
    if target.is_dir() and not allow_directories:
        raise FileOperationSecurityError(
            f"Expected file, got directory: {relative_path}"
        )

    if target.is_file() and allow_directories:
        # This is OK - we allow files when directories are allowed
        pass

    return target


def validate_file_safety(
    base_path: str,
    relative_path: str,
    allowed_extensions: Optional[set[str]] = None,
    max_size_mb: float = MAX_FILE_SIZE_MB,
) -> Path:
    """
    Validate file safety including path, type, and size.

    Args:
        base_path: Base directory path
        relative_path: Relative file path
        allowed_extensions: set of allowed file extensions
        max_size_mb: Maximum file size in MB

    Returns:
        Resolved Path object

    Raises:
        Various security exceptions
    """
    if allowed_extensions is None:
        allowed_extensions = ALLOWED_DOCUMENT_EXTENSIONS

    # Validate path safety first
    filepath = validate_path_safety(base_path, relative_path, allow_directories=False)

    # Validate file extension
    filename = filepath.name
    if not is_safe_extension(filename, allowed_extensions):
        raise FileTypeError(f"File type not allowed: {filename}")

    # Validate file size
    if max_size_mb > 0:
        size_mb = filepath.stat().st_size / (1024 * 1024)
        if size_mb > max_size_mb:
            raise FileSizeError(
                f"File too large: {size_mb:.2f}MB exceeds limit of {max_size_mb}MB"
            )

    return filepath


def validate_directory_safety(
    base_path: str,
    relative_path: str = "",
) -> Path:
    """
    Validate directory safety.

    Args:
        base_path: Base directory path
        relative_path: Relative directory path (empty string for base)

    Returns:
        Resolved Path object

    Raises:
        Various security exceptions
    """
    if relative_path:
        dirpath = validate_path_safety(base_path, relative_path, allow_directories=True)
    else:
        base = Path(base_path).resolve()
        if not base.exists() or not base.is_dir():
            raise DirectoryAccessError(f"Invalid base directory: {base_path}")
        dirpath = base

    return dirpath


# ===========================================================================
# BATCH FILE OPERATIONS
# ===========================================================================


def validate_file_batch(
    base_path: str,
    file_paths: list[str],
    allowed_extensions: Optional[set[str]] = None,
    max_file_size_mb: float = MAX_FILE_SIZE_MB,
    max_batch_size_mb: float = MAX_BATCH_TOTAL_SIZE_MB,
) -> list[Path]:
    """
    Validate a batch of files for security and size constraints.

    Args:
        base_path: Base directory path
        file_paths: List of relative file paths
        allowed_extensions: set of allowed extensions
        max_file_size_mb: Maximum size per file
        max_batch_size_mb: Maximum total batch size

    Returns:
        List of validated Path objects

    Raises:
        FileOperationSecurityError: If any file fails validation
    """
    validated_paths = []
    total_size_mb = 0

    for file_path in file_paths:
        try:
            filepath = validate_file_safety(
                base_path, file_path, allowed_extensions, max_file_size_mb
            )

            size_mb = filepath.stat().st_size / (1024 * 1024)
            total_size_mb += size_mb

            if max_batch_size_mb > 0 and total_size_mb > max_batch_size_mb:
                raise FileSizeError(
                    f"Batch size limit exceeded: {total_size_mb:.2f}MB > {max_batch_size_mb}MB"
                )

            validated_paths.append(filepath)

        except FileOperationSecurityError as e:
            logger.error("File validation failed for %s: %s", file_path, e)
            raise

    return validated_paths


# ===========================================================================
# DIRECTORY LISTING WITH SECURITY
# ===========================================================================


def list_files_safe(
    base_path: str,
    recursive: bool = True,
    allowed_extensions: Optional[set[str]] = None,
    max_file_size_mb: float = MAX_FILE_SIZE_MB,
) -> list[dict]:
    """
    List files in directory with security filtering.

    Args:
        base_path: Base directory path
        recursive: Whether to recurse into subdirectories
        allowed_extensions: set of allowed extensions
        max_file_size_mb: Maximum file size to include

    Returns:
        List of file metadata dicts

    Raises:
        DirectoryAccessError: If base directory is invalid
    """
    if allowed_extensions is None:
        allowed_extensions = ALLOWED_DOCUMENT_EXTENSIONS

    base = validate_directory_safety(base_path)

    files = []
    skipped_large = 0
    skipped_symlinks = 0
    skipped_extension = 0

    pattern = "**/*" if recursive else "*"

    for filepath in base.glob(pattern):
        if not filepath.is_file():
            continue

        # Skip symlinks
        if filepath.is_symlink():
            skipped_symlinks += 1
            continue

        # Check extension
        filename = filepath.name
        if not is_safe_extension(filename, allowed_extensions):
            skipped_extension += 1
            continue

        # Check size
        size_mb = filepath.stat().st_size / (1024 * 1024)
        if max_file_size_mb > 0 and size_mb > max_file_size_mb:
            skipped_large += 1
            continue

        # Get relative path
        try:
            rel_path = filepath.relative_to(base)
            files.append(
                {
                    "key": str(rel_path),
                    "path": filepath,
                    "size": filepath.stat().st_size,
                    "size_mb": size_mb,
                }
            )
        except ValueError:
            logger.warning("Could not get relative path for %s", filepath)
            continue

    logger.info(
        "Found %d files (skipped %d symlinks, %d extensions, %d large)",
        len(files),
        skipped_symlinks,
        skipped_extension,
        skipped_large,
    )

    return files


# ===========================================================================
# RATE LIMITING FOR FILE OPERATIONS
# ===========================================================================


class FileOperationRateLimiter:
    """Rate limiter for file operations per directory."""

    def __init__(self):
        self._operation_counts = {}  # dir_path -> [timestamps]
        self._lock = threading.Lock()

    def is_rate_limited(
        self,
        directory: str,
        max_ops: int = FILE_OPERATION_RATE_LIMIT,
        window_seconds: int = FILE_OPERATION_WINDOW_SECONDS,
    ) -> bool:
        """
        Check if directory has exceeded operation rate limit.

        Args:
            directory: Directory path
            max_ops: Max operations allowed
            window_seconds: Time window in seconds

        Returns:
            True if rate limited, False if operation allowed
        """
        with self._lock:
            current_time = time.time()
            window_start = current_time - window_seconds

            if directory not in self._operation_counts:
                self._operation_counts[directory] = []

            # Remove old entries outside the window
            self._operation_counts[directory] = [
                ts for ts in self._operation_counts[directory] if ts > window_start
            ]

            # Check if exceeded limit
            if len(self._operation_counts[directory]) >= max_ops:
                logger.warning(
                    "Rate limit exceeded for directory %s: %d ops in %d seconds",
                    directory,
                    len(self._operation_counts[directory]),
                    window_seconds,
                )
                return True

            # Record this operation
            self._operation_counts[directory].append(current_time)
            return False


# Global rate limiter instance
_rate_limiter = FileOperationRateLimiter()


def check_file_operation_rate_limit(
    directory: str,
    max_ops: int = FILE_OPERATION_RATE_LIMIT,
) -> bool:
    """
    Check if file operation should be rate limited.

    Args:
        directory: Directory being accessed
        max_ops: Max operations allowed per minute

    Returns:
        True if NOT rate limited (operation allowed), False if rate limited
    """
    is_limited = _rate_limiter.is_rate_limited(
        directory, max_ops, FILE_OPERATION_WINDOW_SECONDS
    )
    return not is_limited


# ===========================================================================
# SECURE FILE READING
# ===========================================================================


def read_file_safe(
    base_path: str,
    relative_path: str,
    allowed_extensions: Optional[set[str]] = None,
    max_size_mb: float = MAX_FILE_SIZE_MB,
) -> bytes:
    """
    Securely read file with all safety checks.

    Args:
        base_path: Base directory path
        relative_path: Relative file path
        allowed_extensions: Allowed file extensions
        max_size_mb: Maximum file size

    Returns:
        File contents as bytes

    Raises:
        Various security exceptions
    """
    filepath = validate_file_safety(
        base_path, relative_path, allowed_extensions, max_size_mb
    )

    # Check rate limit
    base_dir = Path(base_path).resolve()
    if not check_file_operation_rate_limit(str(base_dir)):
        raise FileOperationSecurityError("File operation rate limit exceeded")

    try:
        with open(filepath, "rb") as f:
            return f.read()
    except (IOError, OSError) as e:
        logger.error("Failed to read file %s: %s", filepath, e)
        raise FileOperationSecurityError(f"Cannot read file: {relative_path}") from e


def read_file_text_safe(
    base_path: str,
    relative_path: str,
    encoding: str = "utf-8",
    allowed_extensions: Optional[set[str]] = None,
    max_size_mb: float = MAX_FILE_SIZE_MB,
) -> str:
    """
    Securely read text file.

    Args:
        base_path: Base directory path
        relative_path: Relative file path
        encoding: Text encoding (default: utf-8)
        allowed_extensions: Allowed file extensions
        max_size_mb: Maximum file size

    Returns:
        File contents as string

    Raises:
        Various security exceptions
    """
    content = read_file_safe(base_path, relative_path, allowed_extensions, max_size_mb)

    try:
        return content.decode(encoding)
    except UnicodeDecodeError:
        # Fallback to latin-1 which accepts all bytes
        logger.warning("UTF-8 decode failed for %s, using latin-1", relative_path)
        return content.decode("latin-1", errors="ignore")


# ===========================================================================
# VALIDATION SUMMARY
# ===========================================================================


def get_validation_summary(base_path: str, relative_path: str) -> dict:
    """
    Get detailed validation information about a file.

    Args:
        base_path: Base directory path
        relative_path: Relative file path

    Returns:
        Dictionary with validation details
    """
    try:
        filepath = validate_path_safety(
            base_path, relative_path, allow_directories=False
        )

        filename = filepath.name
        ext = filename.rsplit(".", 1)[1].lower() if "." in filename else ""
        size_mb = filepath.stat().st_size / (1024 * 1024)

        return {
            "path": relative_path,
            "exists": True,
            "is_symlink": filepath.is_symlink(),
            "filename": filename,
            "extension": ext,
            "size_bytes": filepath.stat().st_size,
            "size_mb": size_mb,
            "extension_safe": is_safe_extension(filename),
            "is_within_base": True,
        }
    except FileOperationSecurityError as e:
        return {
            "path": relative_path,
            "exists": False,
            "error": str(e),
        }
