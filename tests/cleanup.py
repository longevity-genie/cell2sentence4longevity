"""Test utilities and helper functions."""

import shutil
import sys
from pathlib import Path
from datetime import datetime, timedelta


def cleanup_old_tests(days: int = 7) -> None:
    """Clean up old test directories.
    
    Args:
        days: Remove test directories older than this many days (0 = all)
    """
    project_root = Path(__file__).parent.parent.parent
    
    print("="*50)
    print("Cleaning up test directories")
    print("="*50)
    print(f"\nRemoving test directories older than {days} days...\n")
    
    cutoff_time = datetime.now() - timedelta(days=days)
    removed_count = 0
    
    # Directories to check
    check_dirs = [
        project_root / 'data' / 'input',
        project_root / 'data' / 'interim',
        project_root / 'data' / 'output',
        project_root / 'logs',
    ]
    
    for base_dir in check_dirs:
        if not base_dir.exists():
            continue
            
        for test_dir in base_dir.glob('test_*'):
            if not test_dir.is_dir():
                continue
                
            # Check modification time
            mtime = datetime.fromtimestamp(test_dir.stat().st_mtime)
            if mtime < cutoff_time:
                print(f"Removing: {test_dir}")
                shutil.rmtree(test_dir, ignore_errors=True)
                removed_count += 1
    
    print(f"\n✓ Removed {removed_count} old test directories")
    
    # Show remaining test directories
    print("\n" + "="*50)
    print("Remaining test directories:")
    print("="*50 + "\n")
    
    for base_dir in check_dirs:
        if not base_dir.exists():
            continue
            
        test_dirs = sorted(base_dir.glob('test_*'))
        if test_dirs:
            print(f"{base_dir.name}/")
            for test_dir in test_dirs:
                mtime = datetime.fromtimestamp(test_dir.stat().st_mtime)
                size = sum(f.stat().st_size for f in test_dir.rglob('*') if f.is_file())
                size_mb = size / (1024 * 1024)
                print(f"  - {test_dir.name} ({mtime.strftime('%Y-%m-%d %H:%M')}, {size_mb:.1f} MB)")
    
    print()


def main() -> None:
    """Main entry point for cleanup command."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Clean up old test directories from integration tests"
    )
    parser.add_argument(
        '--days',
        type=int,
        default=7,
        help='Remove test directories older than N days (0 = all, default: 7)'
    )
    
    args = parser.parse_args()
    
    try:
        cleanup_old_tests(args.days)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

