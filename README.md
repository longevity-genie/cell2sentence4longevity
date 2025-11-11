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
- **Publication metadata** - Optional CellxGene API lookup for publication info (DOI, title, etc.)
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
uv run preprocess run

# Or specify input file explicitly
uv run preprocess run path/to/file.h5ad --input-dir ./data/input

# Process multiple files in batch mode
uv run preprocess run --batch-mode --input-dir ./data/input
```

### Upload to HuggingFace (Optional)

To upload processed data, configure your HuggingFace token:

```bash
cp .env.template .env
# Edit .env and add: HF_TOKEN=your_token_here
```

Then run with repo ID:

```bash
uv run preprocess run --repo-id "username/dataset-name"
```

Get your HuggingFace token from: https://huggingface.co/settings/tokens

## Testing

```bash
# Run all tests
uv run pytest tests/ -v

# Run integration tests (downloads real data from CZI Science)
uv run pytest tests/test_integration.py::TestIntegrationPipeline -v -s

# Clean up test directories older than 7 days
uv run preprocess cleanup

# Remove all test directories
uv run preprocess cleanup --days 0
```

See `tests/README.md` for more details.

## Usage

### Run Pipeline (One-Step Streaming - Recommended)

The `run` command uses a **one-step streaming approach** that processes everything in a single pass:
- Reads h5ad chunks
- Creates cell sentences
- Extracts age from development_stage
- Splits into train/test (stratified by age)
- Writes directly to output
- **No interim files created!**

```bash
# Auto-detect h5ad file from data/input/
uv run preprocess run

# Batch mode - process all h5ad files in a directory
uv run preprocess run --batch-mode --input-dir ./data/input

# Or specify file and folders
uv run preprocess run /path/to/file.h5ad \
  --output-dir ./data/output \
  --repo-id "username/dataset-name"  # Optional, for upload

# Skip train/test split (produce single parquet dataset)
# Useful when you want HuggingFace users to decide on their own splitting
uv run preprocess run /path/to/file.h5ad \
  --skip-train-test-split \
  --output-dir ./data/output

# Publication metadata lookup is enabled by default
# This adds columns: collection_id, publication_title, publication_doi, publication_contact
# To disable: --no-lookup-publication
uv run preprocess run /path/to/file.h5ad \
  --output-dir ./data/output
```

### Command Options

The `run` command supports many options for fine-tuning:

**Input/Output:**
- `h5ad_path` (optional argument) - Path to h5ad file or directory (auto-detects from `--input-dir` if not provided)
- `--input-dir` - Directory containing input files (default: `./data/input`)
- `--output-dir` / `-o` - Directory for final output files (default: `./data/output`)
- `--interim-dir` - Directory for interim files (default: `./data/interim`)

**Processing Options:**
- `--chunk-size` / `-c` - Number of cells per chunk (default: `10000`)
- `--top-genes` - Number of top expressed genes per cell (default: `2000`)
- `--test-size` - Proportion of data for test set (default: `0.05`)
- `--skip-train-test-split` - Skip train/test split and produce single parquet dataset
- `--batch-mode` - Process all h5ad files in input directory
- `--skip-existing` - Skip datasets that already have output files

**Compression:**
- `--compression` - Compression algorithm: `uncompressed`, `snappy`, `gzip`, `lzo`, `brotli`, `lz4`, `zstd` (default: `zstd`)
- `--compression-level` - Compression level: 1-9 for zstd/gzip, 1-11 for brotli (default: `3`)
- `--use-pyarrow` / `--no-pyarrow` - Use pyarrow backend for parquet writes (default: `True`)

**HGNC Gene Mapping:**
- `--mappers` / `-m` - Path to HGNC mappers pickle file (optional, auto-created if needed)
- `--create-hgnc` - Force creation of HGNC mapper (default: auto-created only if needed)

**HuggingFace Upload:**
- `--repo-id` / `-r` - HuggingFace repository ID (e.g., `username/dataset-name`)
- `--token` / `-t` - HuggingFace API token (can also use `HF_TOKEN` env var)

**Publication Metadata:**
- `--lookup-publication` / `--no-lookup-publication` - Enable/disable CellxGene API lookup (default: enabled)

**Logging:**
- `--log-dir` - Directory for log files, separate log per file (default: `./logs`)

**Other:**
- `--keep-interim` - Keep interim parquet files after processing (default: False, cleaned up to save space)

### Publication Metadata Lookup

For datasets from [CellxGene Discover](https://cellxgene.cziscience.com/), publication metadata lookup is **enabled by default**. This automatically adds publication information:

```bash
# Publication lookup is enabled by default
uv run preprocess run ./data/input/10cc50a0-af80-4fa1-b668-893dd5c0113a.h5ad

# To disable publication lookup
uv run preprocess run ./data/input/file.h5ad --no-lookup-publication

# In batch mode (enabled by default)
uv run preprocess run --batch-mode --input-dir ./data/input
```

This queries the CellxGene API and adds the following columns to your output:
- `collection_id` - CellxGene collection ID
- `publication_title` - Title of the associated publication/collection
- `publication_doi` - DOI of the publication (if available)
- `publication_contact` - Contact name for the publication

**Note:** The API lookup may fail for some datasets if:
- The dataset is not from CellxGene Discover
- The CellxGene API changes
- Network issues occur

When lookup fails, processing continues without publication metadata.

### Run Individual Steps (Legacy Two-Step Approach)

If you need more control, you can run individual steps. Note: This creates interim files.

#### Download Dataset

```bash
# Download with default URL (AIDA dataset)
uv run preprocess download

# Download from custom URL
uv run preprocess download --url https://example.com/dataset.h5ad

# Specify output directory and filename
uv run preprocess download \
  --url https://example.com/dataset.h5ad \
  --input-dir ./data/input \
  --filename custom_name.h5ad

# Force re-download even if file exists
uv run preprocess download --force
```

**Download Command Options:**
- `--url` / `-u` - URL to download dataset from (default: AIDA dataset URL)
- `--input-dir` / `-i` - Directory to save downloaded files (default: `./data/input`)
- `--filename` / `-f` - Optional filename (if not provided, extracted from URL)
- `--force` - Force re-download even if file exists
- `--log-file` / `-l` - Path to eliot log file (optional)

#### Create HGNC Mapper

```bash
# Create HGNC mapper explicitly (usually auto-created when needed)
uv run preprocess hgnc-mapper --interim-dir ./data/interim

# Or force creation during run command
uv run preprocess run --create-hgnc
```

**HGNC Mapper Command Options:**
- `--interim-dir` / `-i` - Directory to save interim files (HGNC mappers) (default: `./data/interim`)
- `--log-file` / `-l` - Path to eliot log file (optional)

#### Upload to HuggingFace

```bash
# Upload a single dataset directory (uses default repo-id if not specified)
uv run preprocess upload \
  --output-dir ./data/output \
  --token $HF_TOKEN

# Or specify custom repo-id
uv run preprocess upload \
  --output-dir ./data/output \
  --repo-id "username/dataset-name" \
  --token $HF_TOKEN

# With custom README file
uv run preprocess upload \
  --output-dir ./data/output \
  --repo-id "username/dataset-name" \
  --readme ./README.md \
  --token $HF_TOKEN

# Or upload during processing (per-file uploads)
uv run preprocess run /path/to/file.h5ad \
  --repo-id "username/dataset-name" \
  --token $HF_TOKEN
```

**Upload Command Options:**
- `--output-dir` / `-o` - Directory containing train/test subdirectories (default: `./data/output`)
- `--repo-id` / `-r` - HuggingFace repository ID (default: `longevity-genie/cell2sentence4longevity-data`)
- `--token` / `-t` - HuggingFace API token (required, can also use `HF_TOKEN` env var)
- `--readme` - Path to README file to include in the dataset (optional)
- `--log-file` / `-l` - Path to eliot log file (optional)

## Batch Processing

Process multiple h5ad files efficiently. See `BATCH_PROCESSING.md` for full details.

```bash
# Process all h5ad files in a directory
uv run preprocess run --batch-mode --input-dir ./data/input

# With upload to HuggingFace (uploads after each successful file)
uv run preprocess run --batch-mode \
  --input-dir ./data/input \
  --repo-id username/my-datasets \
  --token $HF_TOKEN

# Skip files that already have output
uv run preprocess run --batch-mode \
  --input-dir ./data/input \
  --skip-existing
```

Key features:
- **One-step streaming**: No interim files, everything in one pass
- **Memory efficient**: Processes terabytes of data with constant memory usage
- **Error isolation**: Failures in one file don't stop others
- **Per-file logging**: Separate logs for each dataset
- **Batch summary**: Creates `batch_processing_summary.tsv` with timing and status for all files

## Project Structure

```
cell2sentence4longevity/
├── src/cell2sentence4longevity/
│   ├── preprocess.py               # Main CLI with all commands
│   ├── cleanup.py                  # Cleanup utilities
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
# Logs are written to --log-dir (default: ./logs)
# Each dataset gets its own log directory
uv run preprocess run /path/to/file.h5ad --log-dir ./logs

# In batch mode, each file gets its own log subdirectory
uv run preprocess run --batch-mode --log-dir ./logs
```

This creates per-dataset logs:
- `./logs/{dataset_name}/pipeline.json` - Machine-readable structured logs
- `./logs/{dataset_name}/pipeline.log` - Human-readable formatted logs

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
