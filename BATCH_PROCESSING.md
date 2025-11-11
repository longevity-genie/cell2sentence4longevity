# Batch Processing Implementation

## Overview

The CLI now supports batch processing of multiple h5ad files with **one-step streaming conversion**, memory-efficient processing, per-file logging, and error handling.

## Key Features

### 1. **Batch Mode Processing**
- Process multiple h5ad files from a folder automatically
- Each file is processed independently with error isolation
- Failures in one file don't stop processing of others
- Progress tracking and summary reporting

### 2. **One-Step Streaming Conversion** (NEW!)
- **No interim files created** - everything happens in a single streaming pass
- Reads h5ad → creates cell sentences → extracts age → splits train/test → writes output
- Memory efficient: data flows through pipeline without intermediate storage
- Significantly faster than two-step approach

### 3. **Memory Management** (Critical for Terabytes of Data)
- **Automatic garbage collection** after each file
- **No interim files** to clean up (one-step approach)
- **Chunked processing**: Never loads entire datasets into memory
- **Backed h5ad loading**: Uses AnnData's backed mode
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

**One-Step Approach (Current):**
```
data/
├── output/
│   ├── dataset1_name/
│   │   ├── train/
│   │   │   └── chunks/
│   │   │       ├── chunk_0000.parquet
│   │   │       └── ...
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

**Note:** No interim files are created with the one-step approach!

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

# With custom output directory
uv run preprocess run-all --batch-mode \
  --input-dir data/input \
  --output-dir data/output \
  --log-dir logs
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

### Step-by-Step Mode (Legacy, Two-Step Approach)

If you need to run individual steps separately:

```bash
# Step 2: Convert all h5ad files to interim parquet (creates interim files)
uv run preprocess step2-convert-h5ad --batch-mode

# Step 3: Create train/test splits from interim files
uv run preprocess step3-train-test-split --interim-dir data/interim/parquet_chunks
```

**Note:** The `run-all` command now uses the one-step approach, which is faster and more memory efficient.

## Command-Line Options

### New Options

- `--batch-mode`: Enable batch processing of all h5ad files in input directory
- `--log-dir PATH`: Directory for log files (default: `./logs`)
- `--skip-train-test-split`: Skip train/test splitting (writes all data to single directory)

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

1. **One-step streaming**: No interim files created, data flows from h5ad → output directly
2. **Backed h5ad loading**: AnnData backed mode prevents loading entire file into memory
3. **Chunked processing**: Processes data in configurable chunks (default: 10,000 cells)
4. **Forced garbage collection**: `gc.collect()` after each file
5. **File isolation**: Each file processed independently, memory freed between files

## Error Handling

- Per-file try-catch blocks
- Failures logged but don't stop batch processing
- Summary report shows success/failure for each file
- Memory freed even on errors via forced garbage collection

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
PROCESSING: my_dataset
Converting h5ad, extracting age, and creating train/test split in one pass
================================================================================
Processing chunks: 100%|███████████████████| 150/150 [00:15:32<00:00,  6.21s/it]
✓ Conversion complete for my_dataset

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


