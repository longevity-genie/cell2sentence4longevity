# Batch Processing Tests

This document describes the batch processing tests added to the test suite.

## Overview

Comprehensive tests for the batch processing functionality have been added to `tests/test_integration.py`. These tests are **skipped by default** because they:
- Download multiple large h5ad files (hundreds of MB each)
- Take a long time to run (30+ minutes)
- Process real data through the full pipeline

## Test Suite

The `TestBatchProcessing` class contains 5 test methods:

### 1. `test_batch_processing_multiple_files`

Tests the full batch processing workflow with multiple real h5ad files.

**What it tests:**
- Downloads 2 real h5ad datasets from CZI Science
- Processes them using `_process_single_file` (batch processing core)
- Validates each file is processed independently
- Checks that per-file logging works correctly
- Validates train/test splits are created for each dataset
- Ensures error isolation (one file failure doesn't stop others)

**Runtime:** ~30-40 minutes (downloads + processing)

### 2. `test_batch_sanitize_dataset_names`

Tests dataset name sanitization for filesystem safety.

**What it tests:**
- Spaces → underscores
- Special characters (@, #, $, etc.) → underscores
- Multiple consecutive underscores → single underscore
- Leading/trailing underscores removed

**Runtime:** < 1 second

### 3. `test_batch_check_output_exists`

Tests the output existence checking functionality.

**What it tests:**
- `check_output_exists` function correctly identifies existing outputs
- Both train/test split mode and single directory mode
- Parquet files are required for output to be considered existing
- Directory structure validation

**Runtime:** < 1 second

### 4. `test_batch_skip_existing_datasets`

Tests the skip logic for already processed datasets.

**What it tests:**
- Processes a dataset once
- Validates `check_output_exists` returns True after processing
- Ensures skip logic would prevent re-processing
- Tests the `--skip-existing` CLI flag functionality

**Runtime:** ~10-15 minutes (one dataset processing)

### 5. `test_batch_memory_management`

Tests memory management in batch processing.

**What it tests:**
- Monitors memory usage before and after processing
- Validates that `gc.collect()` is called in `_process_single_file`
- Logs memory metrics for manual inspection
- Ensures memory doesn't accumulate between files

**Runtime:** ~10-15 minutes (one dataset processing)

**Note:** This test uses `psutil` which was added as a dev dependency.

## Running Batch Tests

### Skip by default

When running the normal test suite, batch tests are automatically skipped:

```bash
# These will skip batch tests
uv run pytest tests/
uv run pytest tests/test_integration.py
```

### Run explicitly

To run batch tests, use one of these methods:

```bash
# Run all batch tests
uv run pytest tests/test_integration.py::TestBatchProcessing -v -s

# Run specific batch test
uv run pytest tests/test_integration.py::TestBatchProcessing::test_batch_processing_multiple_files -v -s

# Use -k to filter by name
uv run pytest -k test_batch -v -s

# Use pytest markers (note: won't work with @pytest.mark.skip)
uv run pytest -m batch -v -s
```

### Run quick tests only

To run only the quick batch tests (name sanitization, output checking):

```bash
uv run pytest -k "test_batch_sanitize or test_batch_check_output_exists" -v -s
```

## Implementation Details

### Skip Mechanism

All batch tests use `@pytest.mark.skip` decorator:

```python
@pytest.mark.skip(reason="Batch processing test - run explicitly with pytest -m batch or pytest -k test_batch")
def test_batch_processing_multiple_files(self, batch_temp_dirs: dict[str, Path]) -> None:
    ...
```

This ensures they are skipped by default but can be run when explicitly selected.

### Test Fixtures

The `batch_temp_dirs` fixture creates timestamped test directories:
- `data/input/` - Shared across runs (preserves downloads)
- `data/interim/batch_test_YYYYMMDD_HHMMSS/` - Interim files
- `data/output/batch_test_YYYYMMDD_HHMMSS/` - Final outputs
- `logs/batch_test_YYYYMMDD_HHMMSS/` - Test logs

Files are **preserved after tests** for manual exploration.

### Dependencies

Batch tests require:
- `psutil>=6.2.0` - For memory monitoring (added to dev dependencies)
- All standard dependencies

Install with:

```bash
uv sync --extra dev
```

## Test Data

Batch tests use real data from CZI Science:
- **Primary:** `10cc50a0-af80-4fa1-b668-893dd5c0113a.h5ad` (~200 MB)
- **Secondary:** `9deda9ad-6a71-401e-b909-5263919d85f9.h5ad` (~150 MB)

These files are downloaded to `data/input/` and reused across test runs.

## Validation Criteria

Each batch test validates:

1. **File Independence**
   - Each file processed separately
   - Failures isolated (don't stop other files)
   - Per-file logging directories created

2. **Output Structure**
   - Train/test directories created
   - Parquet files exist in correct locations
   - Chunk structure is valid

3. **Memory Management**
   - `gc.collect()` called after each file
   - Memory usage monitored and logged
   - No exponential memory growth

4. **Logging**
   - Per-file logs created (JSON + rendered)
   - Eliot action structure preserved
   - Processing times recorded

5. **Error Handling**
   - Try-catch blocks at file level
   - Errors logged but don't crash batch
   - Summary includes success/failure status

## Integration with CI/CD

These tests are **not** intended for CI/CD pipelines due to:
- Long runtime (30+ minutes)
- Large downloads (hundreds of MB)
- Real data processing requirements

For CI/CD, use the existing `TestIntegrationPipeline` tests which are faster and use a single dataset.

## Clean Up

Test outputs are preserved by default. To clean up:

```bash
# Remove test directories older than 7 days
uv run cleanup-tests

# Remove all batch test directories
rm -rf data/interim/batch_test_*
rm -rf data/output/batch_test_*
rm -rf logs/batch_test_*
```

## Future Enhancements

Potential improvements:
1. Add tests for batch summary TSV generation
2. Test HuggingFace upload in batch mode
3. Test error scenarios (corrupted files, network failures)
4. Add performance benchmarks
5. Test with larger datasets (GB+)

