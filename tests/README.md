# Tests

This directory contains integration and unit tests for the cell2sentence4longevity preprocessing pipeline.

## Running Tests

### Install test dependencies

```bash
uv sync --extra dev
```

### Run unit tests (quick)

```bash
uv run pytest tests/test_integration.py::TestLogging tests/test_integration.py::TestAgeExtraction -v
```

### Run integration tests (downloads real data)

```bash
uv run pytest tests/test_integration.py::TestIntegrationPipeline -v -s
```

### Run all tests

```bash
uv run pytest tests/ -v -s
```

### Clean up old test directories

Test output is preserved for manual exploration but can be cleaned up:

```bash
# Remove test directories older than 7 days (default)
uv run cleanup-tests

# Remove test directories older than N days
uv run cleanup-tests --days 3

# Remove all test directories
uv run cleanup-tests --days 0
```

## Test Structure

### Integration Tests (`test_integration.py`)

These tests download real data from CZI Science and run the full preprocessing pipeline:

1. **`test_full_pipeline_with_real_data`**
   - Downloads a real h5ad dataset from https://datasets.cellxgene.cziscience.com/
   - Runs all pipeline steps (create HGNC mapper, convert h5ad, add age, train/test split)
   - Validates that nothing crashes
   - Checks that logs are properly written (both JSON and rendered formats)
   - Validates that age field is numeric (Int64)
   - Checks that age values are reasonable (0-150 years)
   - Verifies output files exist and have correct structure
   - Validates train/test split ratio

2. **`test_pipeline_with_small_chunk_size`**
   - Tests the pipeline with smaller chunk sizes
   - Verifies chunking works correctly
   - Validates chunk sizes are approximately correct

3. **`TestLogging`** - Tests for logging functionality
   - `test_eliot_logging_structure` - Validates eliot log structure
   - `test_log_file_creation` - Tests JSON and rendered log file creation

4. **`TestAgeExtraction`** - Tests for age extraction
   - `test_age_extraction_from_development_stage` - Tests age parsing from various formats
   - `test_age_field_numeric_dtype` - Validates age field has numeric dtype

## Test Data

The integration tests download real data from:
- **URL**: https://datasets.cellxgene.cziscience.com/9deda9ad-6a71-401e-b909-5263919d85f9.h5ad
- **Format**: h5ad (AnnData format)
- **Source**: CZI Science CELLxGENE database

The tests use the project's standard directory structure with timestamped subdirectories:
- `data/input/test_YYYYMMDD_HHMMSS/` - Downloaded h5ad files
- `data/interim/test_YYYYMMDD_HHMMSS/` - Intermediate parquet chunks and HGNC mappers
- `data/output/test_YYYYMMDD_HHMMSS/` - Final train/test splits
- `logs/test_YYYYMMDD_HHMMSS/` - Test logs

**Files are preserved after tests** so you can manually explore the results.

## Notes

- Integration tests may take several minutes to run as they download and process real data
- **Test output files are preserved** in the standard `data/` directories with `test_*` prefixes
- The test will print the exact directory locations where files are stored
- All tests validate proper logging using eliot
- Age field validation ensures numeric type (Int64) and reasonable values

