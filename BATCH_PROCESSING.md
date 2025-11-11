# Batch Processing Implementation

## Overview

The CLI now supports batch processing of multiple h5ad files with memory-efficient processing, per-file logging, error handling, and automatic cleanup of interim files.

## Key Features

### 1. **Batch Mode Processing**
- Process multiple h5ad files from a folder automatically
- Each file is processed independently with error isolation
- Failures in one file don't stop processing of others
- Progress tracking and summary reporting

### 2. **Memory Management** (Critical for Terabytes of Data)
- **Automatic garbage collection** after each processing step
- **Interim file cleanup** (disabled with `--keep-interim` flag)
- Cleans up parquet chunks after creating train/test splits
- Cleans up partial files on errors
- Memory is freed between processing different h5ad files

### 3. **Dataset Name Sanitization**
Each dataset gets its own folder using a sanitized version of the filename:
- Spaces → underscores
- Special characters (@, #, $, etc.) → underscores
- Multiple consecutive underscores → single underscore
- Leading/trailing underscores removed

Example:
```
"my dataset.h5ad"        → "my_dataset"
"dataset@special.h5ad"   → "dataset_special"
"data  with   spaces"    → "data_with_spaces"
```

### 4. **Per-File Directory Structure**
```
data/
├── interim/
│   └── parquet_chunks/
│       ├── dataset1_name/
│       │   └── chunks/
│       │       ├── chunk_0000.parquet
│       │       └── ...
│       └── dataset2_name/
│           └── chunks/
│               └── ...
├── output/
│   ├── dataset1_name/
│   │   ├── train/
│   │   │   └── chunks/
│   │   └── test/
│   │       └── chunks/
│   └── dataset2_name/
│       └── ...
└── logs/
    ├── dataset1_name/
    │   └── pipeline.log
    └── dataset2_name/
        └── pipeline.log
```

### 5. **Per-File Logging**
- Each dataset gets its own log directory
- JSON and rendered log files
- Tracks processing success/failure per file
- Logs cleanup operations

### 6. **HuggingFace Upload (Optional)**
- Upload after each successful file processing
- Dataset name appended to repo_id for multiple files
- Example: `username/repo-dataset1_name`, `username/repo-dataset2_name`

## Usage Examples

### Single File Processing (Traditional)
```bash
# Process one file
uv run preprocess run-all data/input/my_dataset.h5ad

# With options
uv run preprocess run-all data/input/my_dataset.h5ad \
  --output-dir data/output \
  --chunk-size 10000 \
  --keep-interim
```

### Batch Mode Processing
```bash
# Process all h5ad files in a directory
uv run preprocess run-all --batch-mode --input-dir data/input

# Or provide directory as argument
uv run preprocess run-all data/input --batch-mode

# With memory-efficient cleanup (default)
uv run preprocess run-all --batch-mode \
  --input-dir data/input \
  --output-dir data/output \
  --log-dir logs

# Keep interim files (for debugging)
uv run preprocess run-all --batch-mode \
  --input-dir data/input \
  --keep-interim
```

### With HuggingFace Upload
```bash
# Upload each processed dataset
uv run preprocess run-all --batch-mode \
  --input-dir data/input \
  --repo-id username/my-datasets \
  --token $HF_TOKEN

# Results in:
# - username/my-datasets-dataset1
# - username/my-datasets-dataset2
# - ...
```

### Step-by-Step with Batch Mode
```bash
# Step 2: Convert all h5ad files
uv run preprocess step2-convert-h5ad --batch-mode

# Step 3: Add age to all datasets
uv run preprocess step3-add-age --batch-mode

# Step 4: Create splits for all
# (requires manual iteration currently)
```

## Command-Line Options

### New Options

- `--batch-mode`: Enable batch processing of all h5ad files in input directory
- `--keep-interim`: Keep interim parquet files after processing (default: False, cleanup to save space)
- `--log-dir PATH`: Directory for log files (default: `./logs`)

### Existing Options

- `--input-dir PATH`: Directory containing h5ad files (default: `./data/input`)
- `--interim-dir PATH`: Directory for interim processing files (default: `./data/interim`)
- `--output-dir PATH`: Directory for final outputs (default: `./data/output`)
- `--chunk-size INTEGER`: Number of cells per chunk (default: 10000)
- `--top-genes INTEGER`: Number of top genes per cell (default: 2000)
- `--compression TEXT`: Compression algorithm (default: zstd)
- `--test-size FLOAT`: Test set proportion (default: 0.05)
- `--skip-train-test-split`: Skip train/test splitting

## Memory Efficiency Guarantees

For processing terabytes of data, the implementation ensures:

1. **Lazy loading**: Polars `scan_parquet` for streaming operations
2. **Chunked processing**: Never loads entire datasets into memory
3. **Forced garbage collection**: `gc.collect()` after each step
4. **Interim cleanup**: Removes large intermediate files after use
5. **File isolation**: Each file processed independently, memory freed between files

## Error Handling

- Per-file try-catch blocks
- Failures logged but don't stop batch processing
- Partial interim files cleaned up on errors (unless `--keep-interim`)
- Summary report shows success/failure for each file

## Example Output

```
================================================================================
STEP 1: Creating HGNC mapper
================================================================================
✓ Step 1 complete

================================================================================
PROCESSING FILE 1/3: my_dataset
================================================================================

================================================================================
STEP 2: Converting my_dataset to parquet
================================================================================
✓ Step 2 complete for my_dataset

================================================================================
STEP 3: Adding age and cleaning up my_dataset
================================================================================
✓ Step 3 complete for my_dataset

================================================================================
STEP 4: Creating train/test split for my_dataset
================================================================================
✓ Step 4 complete for my_dataset

Cleaning up interim files for my_dataset...
✓ Interim files cleaned up

================================================================================
PROCESSING FILE 2/3: another_dataset
================================================================================
...

================================================================================
PIPELINE COMPLETE - SUMMARY
================================================================================
Total files: 3
Successful: 2
Failed: 1

Details:
  ✓ my_dataset: Successfully processed my_dataset
  ✓ another_dataset: Successfully processed another_dataset
  ✗ broken_dataset: Failed to process broken_dataset: Invalid h5ad format

Output directories:
  Input: data/input
  Interim: data/interim
  Output: data/output
  Logs: logs
```

## Integration with Existing Code

All existing functionality is preserved:
- Single file processing still works as before
- All individual step commands (step1, step2, etc.) still work
- Backward compatible with existing scripts and workflows

