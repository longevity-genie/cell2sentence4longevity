#!/usr/bin/env python3
"""
Demonstration script showing file size limit functionality.

This script simulates the file size checking logic used in batch processing.
"""

from pathlib import Path


def check_file_size_limit(file_path: Path, max_size_mb: float) -> tuple[bool, float, str]:
    """
    Check if a file exceeds the size limit.
    
    Args:
        file_path: Path to the file
        max_size_mb: Maximum allowed file size in MB
        
    Returns:
        Tuple of (should_process, file_size_mb, message)
    """
    # Get file size in MB
    file_size_bytes = file_path.stat().st_size
    file_size_mb = file_size_bytes / (1024 * 1024)
    
    # Check against limit
    if file_size_mb > max_size_mb:
        message = f"File too large: {file_size_mb:.2f} MB > {max_size_mb} MB"
        return False, file_size_mb, message
    else:
        message = f"File size OK: {file_size_mb:.2f} MB <= {max_size_mb} MB"
        return True, file_size_mb, message


def demo_batch_processing_with_size_limit(input_dir: Path, max_file_size_mb: float) -> None:
    """
    Demonstrate batch processing with file size limits.
    
    Args:
        input_dir: Directory containing h5ad files
        max_file_size_mb: Maximum file size in MB
    """
    print(f"Scanning directory: {input_dir}")
    print(f"File size limit: {max_file_size_mb} MB ({max_file_size_mb / 1024:.2f} GB)")
    print("=" * 80)
    
    # Find h5ad files
    h5ad_files = sorted(list(input_dir.glob("*.h5ad")))
    
    if not h5ad_files:
        print(f"No h5ad files found in {input_dir}")
        return
    
    print(f"\nFound {len(h5ad_files)} h5ad file(s):\n")
    
    # Process each file
    processed = 0
    skipped_size = 0
    
    for idx, h5ad_file in enumerate(h5ad_files, 1):
        should_process, file_size_mb, message = check_file_size_limit(h5ad_file, max_file_size_mb)
        
        status = "✓ PROCESS" if should_process else "✗ SKIP"
        print(f"[{idx}/{len(h5ad_files)}] {status} - {h5ad_file.name}")
        print(f"     Size: {file_size_mb:.2f} MB ({file_size_mb / 1024:.2f} GB)")
        print(f"     {message}")
        print()
        
        if should_process:
            processed += 1
        else:
            skipped_size += 1
    
    # Summary
    print("=" * 80)
    print(f"Summary:")
    print(f"  Total files: {len(h5ad_files)}")
    print(f"  Would process: {processed}")
    print(f"  Would skip (size limit): {skipped_size}")
    
    if skipped_size > 0:
        print(f"\n💡 Tip: Increase --max-file-size-mb to process larger files")
    if processed > 0:
        print(f"\n✓ {processed} file(s) are within the size limit and would be processed")


def main() -> None:
    """Main function."""
    import sys
    
    # Example: Check data/input directory with 12 GB limit
    input_dir = Path("./data/input")
    max_file_size_mb = 12000.0  # 12 GB
    
    if len(sys.argv) > 1:
        input_dir = Path(sys.argv[1])
    
    if len(sys.argv) > 2:
        max_file_size_mb = float(sys.argv[2])
    
    if not input_dir.exists():
        print(f"Error: Directory not found: {input_dir}")
        print("\nUsage: python demo_file_size_limit.py [input_dir] [max_size_mb]")
        print("Example: python demo_file_size_limit.py ./data/input 12000")
        sys.exit(1)
    
    demo_batch_processing_with_size_limit(input_dir, max_file_size_mb)


if __name__ == "__main__":
    main()

