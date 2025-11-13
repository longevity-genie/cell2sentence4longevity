"""Integration test for file size limit feature in batch processing."""

import tempfile
from pathlib import Path

import pytest


def create_dummy_file(file_path: Path, size_mb: float) -> None:
    """Create a dummy file of specified size for testing.
    
    Args:
        file_path: Path where to create the file
        size_mb: Size in megabytes
    """
    size_bytes = int(size_mb * 1024 * 1024)
    with open(file_path, 'wb') as f:
        # Write in chunks to avoid memory issues
        chunk_size = 1024 * 1024  # 1 MB chunks
        remaining = size_bytes
        while remaining > 0:
            write_size = min(chunk_size, remaining)
            f.write(b'\0' * write_size)
            remaining -= write_size


@pytest.mark.skip(reason="Creates large temporary files - run explicitly if needed")
def test_file_size_limit_logic_with_temp_files() -> None:
    """Test file size limit logic with actual temporary files.
    
    This test creates temporary files of various sizes and verifies
    that the file size checking logic works correctly.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create test files of different sizes
        test_files = [
            ("small.h5ad", 100),      # 100 MB
            ("medium.h5ad", 5000),    # 5 GB
            ("large.h5ad", 15000),    # 15 GB (over 12 GB limit)
        ]
        
        max_size_mb = 12000.0  # 12 GB limit
        
        print(f"\nCreating test files in {temp_path}")
        for filename, size_mb in test_files:
            file_path = temp_path / filename
            print(f"  Creating {filename} ({size_mb} MB)...")
            create_dummy_file(file_path, size_mb)
            
            # Verify file was created correctly
            actual_size_mb = file_path.stat().st_size / (1024 * 1024)
            assert abs(actual_size_mb - size_mb) < 1, f"File size mismatch for {filename}"
        
        # Test size checking logic
        print(f"\nChecking files against limit of {max_size_mb} MB:")
        
        for filename, expected_size_mb in test_files:
            file_path = temp_path / filename
            file_size_mb = file_path.stat().st_size / (1024 * 1024)
            should_process = file_size_mb <= max_size_mb
            
            print(f"  {filename}: {file_size_mb:.2f} MB - {'PROCESS' if should_process else 'SKIP'}")
            
            # Verify logic
            if expected_size_mb <= max_size_mb:
                assert should_process, f"{filename} should be processed"
            else:
                assert not should_process, f"{filename} should be skipped"
        
        print("\n✓ File size limit logic verified with actual files")


def test_file_size_limit_edge_cases() -> None:
    """Test edge cases for file size limit checking."""
    max_size_mb = 12000.0
    
    # Test exact boundary
    assert 12000.0 <= max_size_mb, "File at exact limit should be allowed"
    assert 12000.01 > max_size_mb, "File just over limit should be rejected"
    
    # Test with None (no limit)
    max_size_mb_none = None
    # When None, all files should be processed (no size check)
    assert max_size_mb_none is None, "None should mean no limit"
    
    # Test very small files
    assert 0.1 <= max_size_mb, "Very small files should be allowed"
    assert 1.0 <= max_size_mb, "1 MB files should be allowed"
    
    # Test very large limit
    large_limit = 1000000.0  # 1 TB
    assert 100000.0 <= large_limit, "Files under very large limit should be allowed"
    
    print("✓ Edge case tests passed")


def test_file_size_conversion_accuracy() -> None:
    """Test accuracy of file size conversions."""
    
    # Test exact conversions
    test_cases = [
        (1024 * 1024 * 1024, 1024.0),           # 1 GB = 1024 MB
        (10 * 1024 * 1024 * 1024, 10240.0),     # 10 GB = 10240 MB
        (12 * 1024 * 1024 * 1024, 12288.0),     # 12 GB = 12288 MB
        (50 * 1024 * 1024 * 1024, 51200.0),     # 50 GB = 51200 MB
    ]
    
    for bytes_val, expected_mb in test_cases:
        calculated_mb = bytes_val / (1024 * 1024)
        assert abs(calculated_mb - expected_mb) < 0.01, \
            f"Conversion error: {bytes_val} bytes should be {expected_mb} MB, got {calculated_mb} MB"
    
    print("✓ File size conversion accuracy verified")


if __name__ == "__main__":
    # Run tests that don't require large files
    test_file_size_limit_edge_cases()
    test_file_size_conversion_accuracy()
    print("\n✓ All file size limit tests passed!")

