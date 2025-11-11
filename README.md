# 🧬 Cell2Sentence4Longevity

[![Tests](https://github.com/longevity-genie/cell2sentence4longevity/actions/workflows/tests.yml/badge.svg)](https://github.com/longevity-genie/cell2sentence4longevity/actions/workflows/tests.yml)

A preprocessing pipeline that converts h5ad single-cell RNA-seq files into "cell sentences" - space-separated lists of gene symbols ordered by expression level for machine learning applications.

## Features

- **One-step streaming** - Reads h5ad → creates cell sentences → extracts age → splits train/test → writes output (no interim files!)
- **Type-safe** - Full type hints throughout
- **Memory-efficient** - Chunked processing with Polars, backed h5ad loading
- **Structured logging** - Eliot logging with detailed diagnostics
- **HGNC gene mapping** - Official gene name conversions (auto-created when needed)
- **Stratified splits** - Train/test splits maintaining age distribution
- **Batch processing** - Process multiple h5ad files in one run
- **Auto-detection** - Finds h5ad files in specified input folder

## Example Dataset

This pipeline was developed for the AIDA (Asian Immune Diversity Atlas) dataset. An example processed dataset is available at:
https://huggingface.co/datasets/transhumanist-already-exists/aida-asian-pbmc-cell-sentence-top2000

The example dataset is not included in this repository. You can download it or use your own h5ad files.

## Installation

```bash
# Install dependencies
uv sync
```

## Quick Start

**One-Step Streaming Approach** (Recommended):

```bash
# Download example AIDA dataset (optional)
uv run preprocess download

# Or use your own h5ad files - place them in data/input/
# The pipeline will auto-detect .h5ad files in the input folder

# Run full pipeline (auto-detects h5ad in data/input/)
# This processes everything in ONE streaming pass: h5ad → cell sentences → age extraction → train/test split → output
uv run preprocess run-all

# Or specify input file explicitly
uv run preprocess run-all path/to/file.h5ad --input-dir ./data/input

# Process multiple files in batch mode
uv run preprocess run-all --batch-mode --input-dir ./data/input
```

### Upload to HuggingFace (Optional)

To upload processed data, configure your HuggingFace token:

```bash
cp .env.template .env
# Edit .env and add: HF_TOKEN=your_token_here
```

Then run with repo ID:

```bash
uv run preprocess run-all --repo-id "username/dataset-name"
```

Get your HuggingFace token from: https://huggingface.co/settings/tokens

## Testing

```bash
# Run all tests
uv run pytest tests/ -v

# Run integration tests (downloads real data from CZI Science)
uv run pytest tests/test_integration.py::TestIntegrationPipeline -v -s

# Clean up test directories older than 7 days
uv run cleanup-tests

# Remove all test directories
uv run cleanup-tests --days 0
```

See `tests/README.md` for more details.

## Usage

### Run All Steps (One-Step Streaming - Recommended)

The `run-all` command uses a **one-step streaming approach** that processes everything in a single pass:
- Reads h5ad chunks
- Creates cell sentences
- Extracts age from development_stage
- Splits into train/test (stratified by age)
- Writes directly to output
- **No interim files created!**

```bash
# Auto-detect h5ad file from data/input/
uv run preprocess run-all

# Batch mode - process all h5ad files in a directory
uv run preprocess run-all --batch-mode --input-dir ./data/input

# Or specify file and folders
uv run preprocess run-all /path/to/file.h5ad \
  --output-dir ./data/output \
  --repo-id "username/dataset-name"  # Optional, for upload

# Skip train/test split (produce single parquet dataset)
# Useful when you want HuggingFace users to decide on their own splitting
uv run preprocess run-all /path/to/file.h5ad \
  --skip-train-test-split \
  --output-dir ./data/output
```

### Run Individual Steps (Legacy Two-Step Approach)

If you need more control, you can run individual steps. Note: This creates interim files.

#### Step 1: Create HGNC Mapper

```bash
uv run preprocess step1-hgnc-mapper --interim-dir ./data/interim
```

#### Step 2: Convert H5AD to Parquet (Two-Step Approach)

**Note:** The `run-all` command now uses the one-step approach which is more efficient.
Use this only if you need separate steps.

```bash
# Auto-detect h5ad from input folder
uv run preprocess step2-convert-h5ad --input-dir ./data/input

# Or specify file explicitly
uv run preprocess step2-convert-h5ad /path/to/file.h5ad \
  --mappers ./data/interim/hgnc_mappers.pkl \
  --interim-dir ./data/interim/parquet_chunks \
  --chunk-size 10000 \
  --top-genes 2000
```

#### Step 3: Create Train/Test Split (Two-Step Approach)

This step can be skipped if you want users on HuggingFace to decide on their own splitting strategy. Use `--skip-train-test-split` flag in the `run-all` command.

```bash
uv run preprocess step3-train-test-split \
  --interim-dir ./data/interim/parquet_chunks \
  --output-dir ./data/output \
  --test-size 0.05 \
  --random-state 42
```

#### Step 4: Upload to HuggingFace

```bash
uv run preprocess step4-upload \
  --output-dir ./data/output \
  --repo-id "username/dataset-name" \
  --token $HF_TOKEN
```

## Batch Processing

Process multiple h5ad files efficiently. See `BATCH_PROCESSING.md` for full details.

```bash
# Process all h5ad files in a directory
uv run preprocess run-all --batch-mode --input-dir ./data/input

# With upload to HuggingFace
uv run preprocess run-all --batch-mode \
  --input-dir ./data/input \
  --repo-id username/my-datasets \
  --token $HF_TOKEN
```

Key features:
- **One-step streaming**: No interim files, everything in one pass
- **Memory efficient**: Processes terabytes of data with constant memory usage
- **Error isolation**: Failures in one file don't stop others
- **Per-file logging**: Separate logs for each dataset

## Project Structure

```
cell2sentence4longevity/
├── src/cell2sentence4longevity/
│   ├── cli.py                      # Main CLI
│   ├── preprocess.py               # Preprocessing CLI (one-step streaming)
│   └── preprocessing/
│       ├── hgnc_mapper.py          # Gene mapping
│       ├── h5ad_converter.py       # H5AD conversion (one-step & two-step)
│       ├── train_test_split.py     # Data splitting (legacy)
│       ├── upload.py               # HuggingFace upload
│       └── download.py             # Dataset download
├── tests/                          # Integration tests
├── docs/                           # Documentation
├── pyproject.toml
└── README.md
```

## Logging

All operations use Eliot structured logging. To enable file logging:

```bash
uv run preprocess run-all /path/to/file.h5ad --log-file ./logs/pipeline.log
```

This creates:
- `./logs/pipeline.json` - Machine-readable structured logs
- `./logs/pipeline.log` - Human-readable formatted logs

### Log Analysis

```bash
# View specific log sections
grep "age_extraction_summary" logs/pipeline.log
grep "filtering_summary" logs/pipeline.log
grep "gene_mapping_summary" logs/pipeline.log
```

See `docs/LOGGING.md` for detailed documentation.

## License

CC BY 4.0
