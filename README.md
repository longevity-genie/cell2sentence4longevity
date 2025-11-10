# Cell2Sentence4Longevity

A preprocessing pipeline that converts h5ad single-cell RNA-seq files into "cell sentences" - space-separated lists of gene symbols ordered by expression level for machine learning applications.

## Features

- **Type-safe** - Full type hints throughout
- **Memory-efficient** - Chunked processing with Polars
- **Structured logging** - Eliot logging with detailed diagnostics
- **HGNC gene mapping** - Official gene name conversions
- **Stratified splits** - Train/test splits maintaining age distribution
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

```bash
# Download example AIDA dataset (optional)
uv run preprocess download

# Or use your own h5ad files - place them in data/input/
# The pipeline will auto-detect .h5ad files in the input folder

# Run full pipeline (auto-detects h5ad in data/input/)
uv run preprocess run-all

# Or specify input folder and file explicitly
uv run preprocess run-all path/to/file.h5ad --input-dir ./data/input
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

### Run All Steps

```bash
# Auto-detect h5ad file from data/input/
uv run preprocess run-all

# Or specify file and folders
uv run preprocess run-all /path/to/file.h5ad \
  --input-dir ./data/input \
  --interim-dir ./data/interim \
  --output-dir ./data/output \
  --repo-id "username/dataset-name"  # Optional, for upload
```

### Run Individual Steps

#### Step 1: Create HGNC Mapper

```bash
uv run preprocess step1-hgnc-mapper --interim-dir ./data/interim
```

#### Step 2: Convert H5AD to Parquet

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

#### Step 3: Add Age and Cleanup

```bash
uv run preprocess step3-add-age --interim-dir ./data/interim/parquet_chunks
```

#### Step 4: Create Train/Test Split

```bash
uv run preprocess step4-train-test-split \
  --interim-dir ./data/interim/parquet_chunks \
  --output-dir ./data/output \
  --test-size 0.05 \
  --random-state 42
```

#### Step 5: Upload to HuggingFace

```bash
uv run preprocess step5-upload \
  --output-dir ./data/output \
  --repo-id "username/dataset-name" \
  --token $HF_TOKEN \
  --max-workers 8
```

## Project Structure

```
cell2sentence4longevity/
├── src/cell2sentence4longevity/
│   ├── cli.py                      # Main CLI
│   ├── preprocess.py               # Preprocessing CLI
│   └── preprocessing/
│       ├── hgnc_mapper.py          # Step 1: Gene mapping
│       ├── h5ad_converter.py       # Step 2: H5AD to parquet
│       ├── age_cleanup.py          # Step 3: Age extraction
│       ├── train_test_split.py     # Step 4: Data splitting
│       ├── upload.py               # Step 5: HuggingFace upload
│       └── download.py             # Dataset download
├── tests/                          # Integration tests
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
