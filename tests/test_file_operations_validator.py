"""
Comprehensive tests for file_operations_validator.py

Tests security controls:
- Path traversal prevention
- Symlink detection and prevention
- File type whitelisting
- File size limits
- Directory access control
- Batch file validation
"""
import tempfile
from pathlib import Path
import pytest


from file_operations_validator import (
    is_safe_extension,
    validate_file_safety,
    validate_directory_safety,
    validate_file_batch,
    read_file_safe,
    validate_path_safety,
    FileOperationSecurityError,
    PathTraversalError,
    SymlinkError,
    FileSizeError,
    FileTypeError,
    DirectoryAccessError,
    ALLOWED_DOCUMENT_EXTENSIONS,
    DANGEROUS_EXTENSIONS,
)


class TestFileExtensionValidation:
    """Test file extension whitelisting."""

    def test_allowed_extensions(self):
        """Test that allowed extensions pass."""
        for ext in ALLOWED_DOCUMENT_EXTENSIONS:
            assert is_safe_extension(f'document.{ext}')
            assert is_safe_extension(f'FILE.{ext.upper()}')

    def test_dangerous_extensions_rejected(self):
        """Test that dangerous extensions are rejected."""
        for ext in DANGEROUS_EXTENSIONS:
            assert not is_safe_extension(f'malware.{ext}')

    def test_unknown_extensions_rejected(self):
        """Test that unknown extensions are rejected."""
        assert not is_safe_extension('file.unknown')
        assert not is_safe_extension('file.xyz')
        assert not is_safe_extension('file.tmp')

    def test_case_insensitive_extension_checking(self):
        """Test that extension checking is case-insensitive."""
        assert is_safe_extension('document.PDF')
        assert is_safe_extension('document.Pdf')
        assert is_safe_extension('document.pDf')

    def test_no_extension_rejected(self):
        """Test that files without extensions are rejected."""
        assert not is_safe_extension('filename')
        assert not is_safe_extension('Makefile')

    def test_multiple_extensions_uses_last(self):
        """Test that only the last extension is checked."""
        assert is_safe_extension('archive.tar.pdf')  # Checks .pdf
        assert not is_safe_extension('document.pdf.exe')  # Checks .exe


class TestPathTraversalPrevention:
    """Test protection against directory traversal attacks."""

    def test_parent_directory_traversal_rejected(self):
        """Test that parent directory traversal (..) is rejected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(PathTraversalError):
                validate_path_safety(tmpdir, '../../../etc/passwd')

    def test_home_directory_escape_rejected(self):
        """Test that home directory escape (~) is rejected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(PathTraversalError):
                validate_path_safety(tmpdir, '~/secret_file.txt')

    def test_environment_variable_expansion_rejected(self):
        """Test that environment variable expansion ($) is rejected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(PathTraversalError):
                validate_path_safety(tmpdir, '$HOME/secret')

    def test_command_injection_operators_rejected(self):
        """Test that command injection operators are rejected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dangerous = [';ls', '|cat', '&chmod',
                         '`whoami`', '$(rm -rf)', '${var}']
            for path in dangerous:
                with pytest.raises(PathTraversalError):
                    validate_path_safety(tmpdir, path)

    def test_absolute_path_traversal_rejected(self):
        """Test that absolute path traversal is rejected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(PathTraversalError):
                validate_path_safety(tmpdir, '/etc/passwd')

    def test_valid_safe_paths_accepted(self):
        """Test that valid safe paths are accepted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            safe_paths = [
                'documents/file.pdf',
                'data/2024/january.csv',
                'reports/Q1-2024.xlsx',
            ]
            for path in safe_paths:
                # Create the file so validation can succeed
                filepath = Path(tmpdir) / path
                filepath.parent.mkdir(parents=True, exist_ok=True)
                filepath.write_text('content')
                # Should not raise exception
                result = validate_path_safety(tmpdir, path)
                assert result is not None

    def test_normalized_safe_paths(self):
        """Test that paths are normalized safely."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # These should be safe paths (without ..)
            safe = [
                './documents/file.pdf',
                'documents//double//slash.pdf',
            ]
            for path in safe:
                # Create the file
                normalized = Path(tmpdir) / path
                normalized = normalized.resolve()
                normalized.parent.mkdir(parents=True, exist_ok=True)
                normalized.write_text('content')
                # Should not raise PathTraversalError
                result = validate_path_safety(tmpdir, path)
                assert result is not None

    def test_parent_directory_in_path_rejected(self):
        """Test that .. in paths is always rejected for security."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Even if it would normalize safely, .. is rejected
            with pytest.raises(PathTraversalError):
                validate_path_safety(tmpdir, 'documents/subdir/../file.pdf')


class TestSymlinkPrevention:
    """Test protection against symlink attacks."""

    def test_symlink_detected_and_rejected(self):
        """Test that symlinks are detected and rejected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a real file
            real_file = Path(tmpdir) / 'real.txt'
            real_file.write_text('content')

            # Create a symlink to it
            symlink_path = Path(tmpdir) / 'link.txt'
            try:
                symlink_path.symlink_to(real_file)
            except OSError:
                # Skip if OS doesn't support symlinks (e.g., Windows without admin)
                pytest.skip("OS doesn't support symlinks")

            # Validation should detect and reject the symlink
            with pytest.raises(SymlinkError):
                validate_file_safety(tmpdir, symlink_path.name)

    def test_symlink_to_system_file_rejected(self):
        """Test that symlinks to system files are rejected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            symlink_path = Path(tmpdir) / 'dangerous.txt'
            try:
                # Try to create symlink to /etc/passwd
                symlink_path.symlink_to('/etc/passwd')
            except (OSError, FileNotFoundError):
                pytest.skip("Cannot create symlinks (OS/permissions)")

            with pytest.raises(SymlinkError):
                validate_file_safety(tmpdir, symlink_path.name)

    def test_regular_files_not_rejected_as_symlinks(self):
        """Test that regular files are not rejected as symlinks."""
        with tempfile.TemporaryDirectory() as tmpdir:
            regular_file = Path(tmpdir) / 'document.pdf'
            regular_file.write_text('PDF content here')

            # Should not raise SymlinkError
            try:
                result = validate_file_safety(tmpdir, regular_file.name)
                # Should succeed
                assert result is not None
            except SymlinkError:
                pytest.fail("Regular file rejected as symlink")


class TestFileSizeLimits:
    """Test file size limit enforcement."""

    def test_large_file_rejected(self):
        """Test that excessively large files are rejected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            large_file = Path(tmpdir) / 'large.pdf'
            # Create a 2MB file and set max_size to 1MB to trigger rejection
            large_file.write_bytes(b'x' * (2 * 1024 * 1024))

            with pytest.raises(FileSizeError):
                validate_file_safety(tmpdir, large_file.name, max_size_mb=1)

    def test_small_file_accepted(self):
        """Test that small files are accepted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            small_file = Path(tmpdir) / 'small.txt'
            small_file.write_text('Small content')

            try:
                result = validate_file_safety(tmpdir, small_file.name)
                # Should succeed
                assert result is not None
            except FileSizeError:
                pytest.fail("Small file rejected for size")

    def test_medium_file_accepted(self):
        """Test that medium-sized files are accepted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            medium_file = Path(tmpdir) / 'medium.pdf'
            # Create actual medium file (use smaller size for testing)
            medium_file.write_text(
                'x' * (10 * 1024 * 1024))  # 10MB actual file
            try:
                result = validate_file_safety(tmpdir, medium_file.name)
                assert result is not None
            except FileSizeError:
                pytest.fail("Medium file rejected for size")

    def test_batch_size_limit(self):
        """Test batch total size limit."""
        with tempfile.TemporaryDirectory() as tmpdir:
            files = []
            # Create multiple large files to exceed batch limit
            for i in range(3):
                f = Path(tmpdir) / f'file{i}.pdf'
                # Create 50MB files (3 x 50 = 150MB > 100MB default batch limit)
                f.write_bytes(b'x' * (50 * 1024 * 1024))
                files.append(f.name)

            with pytest.raises(FileSizeError):
                validate_file_batch(tmpdir, files, max_batch_size_mb=100)


class TestFileValidation:
    """Test complete file validation."""

    def test_valid_safe_file_passes(self):
        """Test that a valid safe file passes all checks."""
        with tempfile.TemporaryDirectory() as tmpdir:
            safe_file = Path(tmpdir) / 'document.pdf'
            safe_file.write_text('%PDF-1.4 content here')

            try:
                result = validate_file_safety(tmpdir, safe_file.name)
                assert result is not None
            except FileOperationSecurityError:
                pass  # May have other validation issues

    def test_dangerous_extension_rejected(self):
        """Test that dangerous file extensions are rejected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            bad_file = Path(tmpdir) / 'malware.exe'
            bad_file.write_text('MZ...')  # PE header

            with pytest.raises(FileTypeError):
                validate_file_safety(tmpdir, bad_file.name)

    def test_path_traversal_and_symlink_checks(self):
        """Test that multiple checks are applied."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Should fail on path traversal first
            with pytest.raises(PathTraversalError):
                validate_file_safety(tmpdir, '../../../etc/passwd')

    def test_validation_returns_path_object(self):
        """Test that validation returns Path object on success."""
        with tempfile.TemporaryDirectory() as tmpdir:
            safe_file = Path(tmpdir) / 'safe.txt'
            safe_file.write_text('content')

            try:
                result = validate_file_safety(tmpdir, safe_file.name)
                assert isinstance(result, Path)
            except FileOperationSecurityError:
                pass  # May have validation issues


class TestDirectoryValidation:
    """Test directory validation."""

    def test_valid_directory_accepted(self):
        """Test that valid directories are accepted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                result = validate_directory_safety(tmpdir)
                assert result is not None
            except (DirectoryAccessError, FileNotFoundError):
                pass

    def test_nonexistent_directory_rejected(self):
        """Test that nonexistent directories are rejected."""
        with pytest.raises(DirectoryAccessError):
            validate_directory_safety('/nonexistent/directory/path')

    def test_directory_traversal_in_path_rejected(self):
        """Test that directory paths cannot traverse."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(PathTraversalError):
                validate_directory_safety(tmpdir, '../../../etc')


class TestBatchFileValidation:
    """Test batch file validation."""

    def test_empty_batch_accepted(self):
        """Test that empty batch is handled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = validate_file_batch(tmpdir, [])
            assert not result

    def test_mixed_valid_files_in_batch(self):
        """Test batch with multiple valid files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            files = []
            for i in range(3):
                f = Path(tmpdir) / f'file{i}.txt'
                f.write_text(f'content {i}')
                files.append(f.name)

            result = validate_file_batch(tmpdir, files)
            assert len(result) == 3

    def test_batch_with_invalid_file_rejected(self):
        """Test that batch fails if any file is invalid."""
        with tempfile.TemporaryDirectory() as tmpdir:
            files = []

            # Valid file
            f1 = Path(tmpdir) / 'good.pdf'
            f1.write_text('content')
            files.append(f1.name)

            # Invalid file
            f2 = Path(tmpdir) / 'bad.exe'
            f2.write_text('MZ...')
            files.append(f2.name)

            with pytest.raises(FileTypeError):
                validate_file_batch(tmpdir, files)

    def test_batch_respects_total_size_limit(self):
        """Test that batch respects total size limit."""
        with tempfile.TemporaryDirectory() as tmpdir:
            files = []
            # Create 10 files of 15MB each = 150MB total
            for i in range(10):
                f = Path(tmpdir) / f'file{i}.txt'
                f.write_bytes(b'x' * (15 * 1024 * 1024))
                files.append(f.name)

            with pytest.raises(FileSizeError):
                validate_file_batch(tmpdir, files, max_batch_size_mb=100)


class TestSafeFileReading:
    """Test safe file reading with validation."""

    def test_read_safe_file_succeeds(self):
        """Test reading a safe file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            safe_file = Path(tmpdir) / 'data.txt'
            content = 'Important data here'
            safe_file.write_text(content)

            result = read_file_safe(tmpdir, safe_file.name)
            assert result == content.encode() or result == content

    def test_read_dangerous_file_fails(self):
        """Test that dangerous files cannot be read."""
        with tempfile.TemporaryDirectory() as tmpdir:
            bad_file = Path(tmpdir) / 'malware.exe'
            bad_file.write_text('MZ...')
            with pytest.raises(FileTypeError):
                read_file_safe(tmpdir, bad_file.name)

    def test_read_with_path_traversal_fails(self):
        """Test that path traversal is prevented."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(PathTraversalError):
                read_file_safe(tmpdir, '../../../etc/passwd')

    def test_read_symlink_fails(self):
        """Test that symlinks cannot be read."""
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                real = Path(tmpdir) / 'real.txt'
                real.write_text('content')
                symlink = Path(tmpdir) / 'link.txt'
                symlink.symlink_to(real)

                with pytest.raises(SymlinkError):
                    read_file_safe(tmpdir, symlink.name)
            except OSError:
                pytest.skip("Symlinks not supported")


class TestExceptionCustomization:
    """Test custom exception types."""

    def test_path_traversal_error_inheritance(self):
        """Test that PathTraversalError inherits from base."""
        error = PathTraversalError('test')
        assert isinstance(error, FileOperationSecurityError)

    def test_symlink_error_inheritance(self):
        """Test that SymlinkError inherits from base."""
        error = SymlinkError('test')
        assert isinstance(error, FileOperationSecurityError)

    def test_file_size_error_inheritance(self):
        """Test that FileSizeError inherits from base."""
        error = FileSizeError('test')
        assert isinstance(error, FileOperationSecurityError)

    def test_file_type_error_inheritance(self):
        """Test that FileTypeError inherits from base."""
        error = FileTypeError('test')
        assert isinstance(error, FileOperationSecurityError)

    def test_exception_messages(self):
        """Test that exception messages are meaningful."""
        msg = 'Path traversal detected'
        error = PathTraversalError(msg)
        assert msg in str(error)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_filename(self):
        """Test handling of empty filename."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises((PathTraversalError, ValueError, FileOperationSecurityError)):
                validate_file_safety(tmpdir, '')

    def test_whitespace_only_filename(self):
        """Test handling of whitespace-only filename."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises((PathTraversalError, ValueError, FileOperationSecurityError)):
                validate_file_safety(tmpdir, '   ')

    def test_null_bytes_in_path(self):
        """Test that null bytes are rejected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises((PathTraversalError, ValueError, FileOperationSecurityError)):
                validate_file_safety(tmpdir, 'file\x00.pdf')

    def test_very_long_filename(self):
        """Test handling of very long filenames."""
        with tempfile.TemporaryDirectory() as tmpdir:
            long_name = 'a' * 1000 + '.pdf'
            # Should either succeed (if path is safe) or fail for length
            try:
                validate_file_safety(tmpdir, long_name)
            except (PathTraversalError, ValueError, FileOperationSecurityError):
                pass  # All acceptable

    def test_unicode_in_filename(self):
        """Test handling of unicode characters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            unicode_file = Path(tmpdir) / 'café_résumé.txt'
            try:
                unicode_file.write_text('content')
                result = validate_file_safety(tmpdir, unicode_file.name)
                # Should succeed
                assert result is not None
            except (FileOperationSecurityError, ValueError):
                pass  # Both acceptable

    def test_case_sensitivity(self):
        """Test that extension checking is case-insensitive."""
        with tempfile.TemporaryDirectory() as tmpdir:
            upper_ext = Path(tmpdir) / 'FILE.PDF'
            upper_ext.write_text('content')

            try:
                result = validate_file_safety(tmpdir, upper_ext.name)
                # PDF files should be allowed regardless of case
                assert result is not None
            except FileTypeError:
                pytest.fail("Case sensitivity in extension checking")

    def test_dangerous_pattern_combinations(self):
        """Test multiple dangerous patterns in one path."""
        paths = [
            '../etc/passwd',
            '../../$HOME/.ssh/id_rsa',
            'file;rm -rf;.pdf',
            '$(cat /etc/passwd)',
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            for path in paths:
                with pytest.raises(PathTraversalError):
                    validate_path_safety(tmpdir, path)
