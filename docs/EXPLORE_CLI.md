# Explore CLI - AnnData Metadata Extraction Tool

A memory-efficient CLI tool for extracting metadata fields from h5ad AnnData files into Polars DataFrames saved as Parquet files.

## Installation

The `explore` command is automatically installed with the project:

```bash
uv sync
```

## Usage

### Single File Extraction

Extract specific metadata fields from a single h5ad file:

```bash
uv run explore extract <h5ad_file> -f <field1> -f <field2> -f <field3>
```

**Example:**
```bash
uv run explore extract data/input/sample.h5ad \
  -f development_stage \
  -f cell_type \
  -f tissue \
  -o data/output/sample_meta.parquet
```

**Options:**
- `--field`, `-f`: Field name to extract (can be specified multiple times) - **required**
- `--output`, `-o`: Output parquet file path (default: `<h5ad_name>_meta.parquet`)
- `--chunk-size`: Number of rows to process at a time (default: 10000)
- `--compression`: Compression algorithm (default: zstd)
- `--compression-level`: Compression level (default: 3)
- `--log-dir`: Directory for log files

### Batch Extraction

Extract metadata fields from multiple h5ad files in a directory:

```bash
uv run explore batch <input_dir> <output_dir> -f <field1> -f <field2>
```

**Example:**
```bash
uv run explore batch data/input data/output/metadata \
  -f development_stage \
  -f cell_type \
  -f tissue \
  -f donor_id \
  --log-dir logs/batch_extract
```

**Options:**
- `--field`, `-f`: Field name to extract (can be specified multiple times) - **required**
- `--chunk-size`: Number of rows to process at a time (default: 10000)
- `--compression`: Compression algorithm (default: zstd)
- `--compression-level`: Compression level (default: 3)
- `--log-dir`: Directory for log files
- `--skip-existing` / `--overwrite`: Skip files that already have output (default: skip-existing)

## Output Format

All output files are saved as Parquet files with the `_meta` suffix:
- Single file: `<input_name>_meta.parquet`
- Batch mode: `<input_name>_meta.parquet` for each input file

The output files contain only the requested metadata fields as columns, extracted from the `adata.obs` DataFrame.

## Memory Efficiency

The tool is designed to be memory-efficient:
1. **Backed mode**: H5ad files are opened in `backed='r'` mode, keeping data on disk
2. **Chunked processing**: Data is processed in chunks (default: 10,000 rows at a time)
3. **Streaming**: Row slices are extracted first before selecting columns
4. **Direct conversion**: Data flows from AnnData → Polars → Parquet without intermediate copies

This allows processing very large h5ad files (millions of cells) on systems with limited RAM.

## Examples

### Extract age-related metadata
```bash
uv run explore extract data.h5ad -f development_stage -f age -f donor_id
```

### Extract cell type annotations
```bash
uv run explore extract data.h5ad \
  -f cell_type \
  -f cell_type_ontology_term_id \
  -f tissue \
  -f organ
```

### Batch extract from multiple datasets
```bash
uv run explore batch ./datasets ./metadata \
  -f development_stage \
  -f cell_type \
  -f tissue \
  -f disease \
  --skip-existing \
  --log-dir ./logs
```

### Check what fields are available
To see what fields are available in your h5ad file, you can use Python:
```python
import anndata as ad
adata = ad.read_h5ad('your_file.h5ad', backed='r')
print(adata.obs.columns.tolist())
adata.file.close()
```

## Logging

The tool uses Eliot for structured logging. When you specify `--log-dir`, it creates:
- `extract.json` / `batch_extract.json`: Machine-readable JSON logs
- `extract.log` / `batch_extract.log`: Human-readable rendered logs

Without `--log-dir`, logs are written to stdout and a JSON file in the current directory.

## Error Handling

- If a requested field doesn't exist, the tool will warn you and extract only the available fields
- In batch mode, errors in one file don't stop processing of other files
- Failed files are reported in the final summary


