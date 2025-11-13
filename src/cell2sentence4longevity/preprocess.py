"""Preprocessing command-line interface for cell2sentence pipeline."""

from pathlib import Path
from typing import Optional
import gc
import shutil
import re
import sys

import typer
from eliot import start_action, to_file
from pycomfort.logging import to_nice_file, to_nice_stdout
from dotenv import load_dotenv

from cell2sentence4longevity.preprocessing import (
    convert_h5ad_to_train_test,
    upload_to_huggingface,
    download_dataset,
)
from cell2sentence4longevity.preprocessing.upload import DEFAULT_REPO_ID
from cell2sentence4longevity.cleanup import cleanup_old_tests

# Load environment variables from .env file
load_dotenv()

app = typer.Typer(
    name="preprocess",
    help="Cell to sentence preprocessing pipeline for longevity research",
    add_completion=False,
)

DATASET_CARD_DEFAULT_PATH = Path("./docs/DATASET_CARD.md")


def sanitize_dataset_name(name: str) -> str:
    """Sanitize dataset name for use as directory name.
    
    - Replace spaces with underscores
    - Replace other problematic characters with underscores
    - Remove consecutive underscores
    - Remove leading/trailing underscores
    
    Args:
        name: Original dataset name
        
    Returns:
        Sanitized dataset name safe for filesystem use
    """
    # Replace spaces and problematic characters with underscores
    sanitized = re.sub(r'[^\w\-.]', '_', name)
    # Remove consecutive underscores
    sanitized = re.sub(r'_+', '_', sanitized)
    # Remove leading/trailing underscores
    sanitized = sanitized.strip('_')
    return sanitized


def check_output_exists(output_dir: Path, dataset_name: str, skip_train_test_split: bool) -> bool:
    """Check if output files already exist for a dataset.
    
    Args:
        output_dir: Base output directory
        dataset_name: Name of the dataset
        skip_train_test_split: Whether train/test split was skipped
        
    Returns:
        True if output exists (dataset folder exists and contains parquet files), False otherwise
    """
    dataset_output_dir = output_dir / dataset_name
    
    if not dataset_output_dir.exists():
        return False
    
    if skip_train_test_split:
        # Check dataset directory directly for parquet files
        if dataset_output_dir.exists():
            parquet_files = list(dataset_output_dir.glob("*.parquet"))
            return len(parquet_files) > 0
        else:
            return False
    else:
        # Check for train and test directories with parquet files (no chunks subfolders)
        train_dir = dataset_output_dir / "train"
        test_dir = dataset_output_dir / "test"
        
        train_parquet_files = list(train_dir.glob("*.parquet")) if train_dir.exists() else []
        test_parquet_files = list(test_dir.glob("*.parquet")) if test_dir.exists() else []
        
        # Return True if at least one parquet file exists in either train or test
        return len(train_parquet_files) > 0 or len(test_parquet_files) > 0


@app.command()
def download(
    url: str = typer.Option(
        "https://datasets.cellxgene.cziscience.com/9deda9ad-6a71-401e-b909-5263919d85f9.h5ad",
        "--url",
        "-u",
        help="URL to download dataset from"
    ),
    input_dir: Path = typer.Option(
        Path("./data/input"),
        "--input-dir",
        "-i",
        help="Directory to save downloaded files"
    ),
    filename: str | None = typer.Option(
        None,
        "--filename",
        "-f",
        help="Optional filename. If not provided, extracted from URL"
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Force re-download even if file exists"
    ),
    log_file: Path | None = typer.Option(
        None,
        "--log-file",
        "-l",
        help="Path to eliot log file"
    ),
    log_stdout: bool = typer.Option(
        True,
        "--log-stdout/--no-log-stdout",
        help="Mirror Eliot logs to stdout"
    ),
) -> None:
    """Download dataset from a URL.
    
    Downloads datasets (typically h5ad files) to the input directory.
    By default, skips download if file already exists. Use --force to re-download.
    """
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        json_path = log_file.with_suffix('.json')
        to_nice_file(output_file=json_path, rendered_file=log_file)
        if log_stdout:
            to_nice_stdout(output_file=json_path)
    elif log_stdout:
        to_file(sys.stdout)
    
    with start_action(action_type="cli_download") as action:
        if force:
            typer.echo(f"Force downloading dataset from {url}...")
            action.log(message_type="download_started", url=url, force=True)
        else:
            typer.echo(f"Downloading dataset from {url} (skipping if already exists)...")
            action.log(message_type="download_started", url=url, force=False)
        output_path = download_dataset(url, input_dir, filename, force=force)
        typer.secho("✓ Download completed successfully", fg=typer.colors.GREEN)
        typer.echo(f"Saved to: {output_path}")
        action.log(message_type="download_completed", output_path=str(output_path))


@app.command()
def cleanup(
    days: int = typer.Option(
        7,
        "--days",
        "-d",
        help="Remove test directories older than N days (0 = all)"
    ),
) -> None:
    """Clean up old test directories.
    
    Removes test directories from data/input, data/interim, data/output, and logs
    that are older than the specified number of days. Use --days 0 to remove all test directories.
    """
    with start_action(action_type="cli_cleanup") as action:
        typer.echo("Cleaning up old test directories...")
        action.log(message_type="cleanup_started", days=days)
        cleanup_old_tests(days)
        typer.secho("✓ Cleanup completed", fg=typer.colors.GREEN)
        action.log(message_type="cleanup_completed", days=days)


@app.command()
def upload(
    output_dir: Path = typer.Option(
        Path("./data/output"),
        "--output-dir",
        "-o",
        help="Directory containing train/test subdirectories"
    ),
    repo_id: str = typer.Option(
        "longevity-genie/cell2sentence4longevity-data",
        "--repo-id",
        "-r",
        help="HuggingFace repository ID (e.g., 'username/dataset-name'). Defaults to 'longevity-genie/cell2sentence4longevity-data'"
    ),
    token: str = typer.Option(
        ...,
        "--token",
        "-t",
        help="HuggingFace API token",
        envvar="HF_TOKEN"
    ),
    readme_path: Optional[Path] = typer.Option(
        None,
        "--readme",
        help="Path to README file"
    ),
    log_file: Optional[Path] = typer.Option(
        None,
        "--log-file",
        "-l",
        help="Path to eliot log file"
    ),
    log_stdout: bool = typer.Option(
        True,
        "--log-stdout/--no-log-stdout",
        help="Mirror Eliot logs to stdout"
    ),
) -> None:
    """Upload to HuggingFace.
    
    Uploads the processed data to HuggingFace in a single batch commit.
    """
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        json_path = log_file.with_suffix('.json')
        to_nice_file(output_file=json_path, rendered_file=log_file)
    if log_stdout:
        to_file(sys.stdout)
    
    with start_action(action_type="cli_upload") as action:
        typer.echo("Uploading to HuggingFace...")
        action.log(message_type="upload_started", repo_id=repo_id, output_dir=str(output_dir))
        files_uploaded = upload_to_huggingface(
            data_splits_dir=output_dir,
            token=token,
            repo_id=repo_id,
            readme_path=readme_path
        )
        dataset_url = f"https://huggingface.co/datasets/{repo_id}"
        if files_uploaded:
            typer.secho("✓ Upload completed successfully", fg=typer.colors.GREEN)
        typer.echo(f"Dataset: {dataset_url}")
        action.log(message_type="upload_completed", repo_id=repo_id, dataset_url=dataset_url, files_uploaded=files_uploaded)


@app.command("dataset-card")
def dataset_card(
    output_path: Path = typer.Option(
        DATASET_CARD_DEFAULT_PATH,
        "--output",
        "-o",
        help="Where to write the dataset card (Markdown). Default: ./docs/DATASET_CARD.md"
    ),
    upload: bool = typer.Option(
        False,
        "--upload/--no-upload",
        help="If set, upload the dataset card to Hugging Face as README.md"
    ),
    repo_id: str = typer.Option(
        "longevity-genie/cell2sentence4longevity-data",
        "--repo-id",
        "-r",
        help="HuggingFace repository ID to upload README to"
    ),
    token: Optional[str] = typer.Option(
        None,
        "--token",
        "-t",
        help="HuggingFace API token (required if --upload is set)",
        envvar="HF_TOKEN"
    ),
    log_file: Optional[Path] = typer.Option(
        None,
        "--log-file",
        "-l",
        help="Path to eliot log file"
    ),
    log_stdout: bool = typer.Option(
        True,
        "--log-stdout/--no-log-stdout",
        help="Mirror Eliot logs to stdout"
    ),
) -> None:
    """Generate the dataset card (Markdown) and optionally upload it as README.md."""
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        json_path = log_file.with_suffix('.json')
        to_nice_file(output_file=json_path, rendered_file=log_file)
        if log_stdout:
            to_nice_stdout(output_file=json_path)
    elif log_stdout:
        to_file(sys.stdout)

    with start_action(action_type="cli_dataset_card", will_upload=upload) as action:
        # Ensure directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Write the dataset card content
        card = f"""## Dataset Card: {repo_id}

### Summary
This repository contains preprocessed single-cell RNA-seq (scRNA‑seq) datasets prepared as “cell sentences” for training and evaluation of cells2sentence-style models. Each cell is represented as a space‑separated sequence of top expressed gene symbols, enabling language‑model style training for tasks such as biological age prediction and other downstream applications.

This dataset targets fine‑tuning and evaluation of models inspired by cells2sentence approaches for cellular phenotyping, including age prediction as described in the preprint: [cells2sentence: Sequence models on gene expression](https://www.biorxiv.org/content/10.1101/2025.04.14.648850v3.full).

### What are “cell sentences”?
For each cell, we rank genes by expression and keep the top N (default 2000). We filter out Ensembl IDs and keep valid gene symbols, then serialize them as a whitespace‑separated string. This converts a numeric high‑dimensional cell profile into a token sequence amenable to language‑model training.

### Supported tasks and use cases
- Age prediction from single‑cell expression profiles
- Tissue/organ classification
- Cell type labeling and transfer
- Condition/disease stratification and dataset harmonization
- Few‑shot or instruction‑style fine‑tuning of sequence models on cells

### Data sources and provenance
- Source data are public scRNA‑seq h5ad datasets, primarily from the CZI CellxGene collections.
- When a dataset is detected as CellxGene (by UUID), we add `dataset_id` and, where available via cached collections metadata, join publication information:
  - `collection_id`, `publication_title`, `publication_doi`, `publication_description`, `publication_contact_name`, `publication_contact_email`.
- The pipeline is streaming and memory‑efficient, and uses Polars for processing.

### Repository structure
Each source dataset is organized under its own subfolder. There are two common layouts:
- Train/test split (default):
  - `<dataset_name>/train/chunk_*.parquet`
  - `<dataset_name>/test/chunk_*.parquet`
- Single split (if train/test split is disabled):
  - `<dataset_name>/chunk_*.parquet` or `<dataset_name>/chunks/chunk_*.parquet`

### Data fields (columns)
Columns are inherited from the input AnnData `.obs` table, plus generated fields:
- `cell_sentence` (string): space‑separated gene symbols for the cell (top‑N expression).
- `age` (float): numeric age extracted from `development_stage` where parsable (years). Cells with null age are filtered by default for training splits.
- `dataset_id` (string, optional): CellxGene dataset UUID when detected.
- Publication fields (optional, when join succeeds): `collection_id`, `publication_title`, `publication_doi`, `publication_description`, `publication_contact_name`, `publication_contact_email`.
- Other `.obs` fields (optional, dataset‑specific): e.g., `organism`, `tissue`, `cell_type`, `assay`, `sex`, `disease`, etc.

Notes:
- In current train/test outputs, the standardized column is `age` (years) when extractable from `development_stage`. Some upstream datasets encode mouse age in months; those may not map into `age` unless present in a parsable “year‑old” format.

### Preparation pipeline (high level)
1. Read h5ad in backed mode (streaming).
2. Map genes to symbols (HGNC lookup where helpful); filter out Ensembl IDs from sentences.
3. Build `cell_sentence` from top expressed genes per cell (default top‑N = 2000).
4. Extract `age` from `development_stage` when available (numeric years).
5. Optionally add `dataset_id` and join publication metadata if the dataset is found in CellxGene collections cache.
6. Filter cells with null `age` by default (for consistent age‑based tasks).
7. Write Parquet chunks and, by default, produce train/test split stratified by `age` (~95/5).

### How to use
Below is an example for downloading the repository snapshot and loading with Polars. This approach is scalable and keeps a local cache.

```python
from pathlib import Path
import polars as pl
from huggingface_hub import snapshot_download

repo_id = "{repo_id}"
local_dir = Path(snapshot_download(repo_id=repo_id, repo_type="dataset"))

# Example: load train split for one dataset folder
dataset_name = "10cc50a0-af80-4fa1-b668-893dd5c0113a"  # replace with any available subfolder
train_glob = local_dir / dataset_name / "train" / "chunk_*.parquet"
test_glob = local_dir / dataset_name / "test" / "chunk_*.parquet"

train_df = pl.scan_parquet(str(train_glob)).collect()
test_df = pl.scan_parquet(str(test_glob)).collect()

# Basic checks
assert "cell_sentence" in train_df.columns
assert "age" in train_df.columns
```

You can iterate across all dataset subfolders to build training mixtures, or concatenate multiple datasets at scan‑time for large‑scale training pipelines.

### Limitations and caveats
- Not all datasets provide a reliably parsable human age; cells with null `age` are filtered for the default split.
- For mouse datasets that encode months (e.g., “24m”), month handling may appear in metadata extraction utilities but train/test outputs standardize on `age` when parsable as years.
- `.obs` schema varies across sources; presence of optional fields is dataset‑dependent.

### Licensing
- This repository aggregates preprocessed derivatives of public scRNA‑seq datasets. The original data remain under their respective licenses (see the source collection pages on CellxGene and corresponding publications). Please respect upstream licensing and citation requirements when using the data.
- The dataset card and pipeline code are provided under the project’s license; data licensing follows the upstream sources.

### Citation
If you use this dataset, please cite:
- cells2sentence preprint: “Sequence models on gene expression.” BioRxiv, 2025. [Link](https://www.biorxiv.org/content/10.1101/2025.04.14.648850v3.full)
- CellxGene data portal and the individual source publications for datasets included in this collection.

### Contact
Maintainer: `longevity-genie` on Hugging Face. Issues and improvements are welcome.
"""
        output_path.write_text(card)
        action.log(message_type="dataset_card_written", path=str(output_path), bytes=len(card.encode("utf-8")))
        typer.secho(f"✓ Dataset card written to: {output_path}", fg=typer.colors.GREEN)

        if upload:
            if not token:
                typer.secho("Error: --upload requires --token (or HF_TOKEN env var).", fg=typer.colors.RED)
                raise typer.Exit(1)
            # Upload README.md only (no data files) using huggingface_hub directly
            action.log(message_type="uploading_readme_only", repo_id=repo_id, readme=str(output_path))
            from huggingface_hub import HfApi, CommitOperationAdd
            api = HfApi(token=token)
            api.create_repo(repo_id=repo_id, repo_type="dataset", private=False, exist_ok=True)
            commit_info = api.create_commit(
                repo_id=repo_id,
                repo_type="dataset",
                operations=[
                    CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=str(output_path))
                ],
                commit_message="Update dataset README"
            )
            dataset_url = f"https://huggingface.co/datasets/{repo_id}"
            typer.secho("✓ README uploaded successfully", fg=typer.colors.GREEN)
            typer.echo(f"Dataset: {dataset_url}")
            action.log(message_type="readme_upload_completed", repo_id=repo_id, dataset_url=dataset_url, commit_url=commit_info.commit_url)


def _process_single_file(
    h5ad_path: Path,
    output_dir: Path,
    chunk_size: int,
    top_genes: Optional[int],
    compression: str,
    compression_level: int,
    use_pyarrow: bool,
    test_size: float,
    skip_train_test_split: bool,
    repo_id: Optional[str],
    token: Optional[str],
    join_collection: bool = True,
    filter_by_age: bool = True,
    gene_lists_dir: Optional[Path] = None,
) -> tuple[bool, str, float, Path]:
    """Process a single h5ad file through the entire pipeline.
    
    This uses the one-step convert_h5ad_to_train_test() function which:
    - Reads h5ad file
    - Creates cell sentences
    - Extracts age from development_stage
    - Splits into train/test (if not skipped)
    - Writes directly to output (no interim files)
    
    Args:
        h5ad_path: Path to the h5ad file
        output_dir: Base output directory
        chunk_size: Number of cells per chunk
        top_genes: Number of top genes per cell
        compression: Compression algorithm
        compression_level: Compression level
        use_pyarrow: Whether to use pyarrow
        test_size: Test set size proportion
        skip_train_test_split: Whether to skip train/test split
        repo_id: HuggingFace repository ID
        token: HuggingFace token
    Returns:
        Tuple of (success: bool, message: str, processing_time_seconds: float, dataset_output_dir: Path)
    """
    import time
    
    dataset_name = sanitize_dataset_name(h5ad_path.stem)
    dataset_output_dir = output_dir / dataset_name
    start_time = time.time()
    
    with start_action(action_type="process_single_file", dataset_name=dataset_name, h5ad_path=str(h5ad_path)) as action:
        try:
            # One-step conversion: h5ad -> cell sentences + age extraction -> train/test split -> output
            action.log(
                message_type="file_processing_started",
                dataset_name=dataset_name,
                chunk_size=chunk_size,
                top_genes=top_genes,
                test_size=test_size,
                skip_train_test_split=skip_train_test_split
            )
            
            convert_h5ad_to_train_test(
                h5ad_path=h5ad_path,
                output_dir=output_dir,
                dataset_name=dataset_name,
                chunk_size=chunk_size,
                top_genes=top_genes,
                test_size=test_size,
                random_state=42,
                compression=compression,
                compression_level=compression_level,
                use_pyarrow=use_pyarrow,
                skip_train_test_split=skip_train_test_split,
                stratify_by_age=True,
                join_collection=join_collection,
                filter_by_age=filter_by_age,
                gene_lists_dir=gene_lists_dir
            )
            
            typer.echo("  ✓ Conversion completed")
            action.log(message_type="conversion_completed", dataset_name=dataset_name)
            
            # Force garbage collection to free memory
            gc.collect()
            
            # Determine upload directory
            if skip_train_test_split:
                upload_dir = output_dir / dataset_name / "chunks"
            else:
                upload_dir = output_dir / dataset_name
            
            # Upload to HuggingFace (optional)
            if repo_id and token:
                # Upload to same repository as subfolder (dataset_name creates subfolder in repo)
                action.log(message_type="upload_started", dataset_name=dataset_name, repo_id=repo_id, upload_dir=str(upload_dir))
                typer.echo(f"  Uploading {dataset_name} to HuggingFace...")
                files_uploaded = upload_to_huggingface(
                    data_splits_dir=upload_dir,
                    token=token,
                    repo_id=repo_id,
                    dataset_name=dataset_name
                )
                dataset_url = f"https://huggingface.co/datasets/{repo_id}"
                if files_uploaded:
                    typer.echo(f"  ✓ Upload completed: {dataset_url}")
                else:
                    typer.echo("  ⚠ Upload returned False")
                action.log(message_type="upload_completed", dataset_name=dataset_name, repo_id=repo_id, dataset_url=dataset_url, files_uploaded=files_uploaded)
            
            # Final garbage collection
            gc.collect()
            
            processing_time = time.time() - start_time
            action.log(message_type="file_processing_success", dataset_name=dataset_name, processing_time_seconds=round(processing_time, 2))
            return True, f"Successfully processed {dataset_name}", processing_time, dataset_output_dir
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Failed to process {dataset_name}: {str(e)}"
            action.log(message_type="file_processing_error", dataset_name=dataset_name, error=str(e), processing_time_seconds=round(processing_time, 2))
            
            # Force garbage collection even on error
            gc.collect()
            
            return False, error_msg, processing_time, dataset_output_dir


@app.command()
def run(
    h5ad_path: Path | None = typer.Argument(
        None,
        help="Path to AIDA h5ad file or directory containing h5ad files (optional, auto-detects from --input-dir if not provided)"
    ),
    input_dir: Path = typer.Option(
        Path("./data/input"),
        "--input-dir",
        help="Directory containing input files (used for auto-detection or as fallback)"
    ),
    output_dir: Path = typer.Option(
        Path("./data/output"),
        "--output-dir",
        "-o",
        help="Directory for final output files"
    ),
    repo_id: Optional[str] = typer.Option(
        DEFAULT_REPO_ID,
        "--repo-id",
        "-r",
        help=f"HuggingFace repository ID (e.g., 'username/dataset-name'). Defaults to '{DEFAULT_REPO_ID}'. For batch processing, all datasets are uploaded to the same repository as subfolders."
    ),
    token: Optional[str] = typer.Option(
        None,
        "--token",
        "-t",
        help="HuggingFace API token (upload after each file if provided)",
        envvar="HF_TOKEN"
    ),
    chunk_size: int = typer.Option(
        10000,
        "--chunk-size",
        "-c",
        help="Number of cells per chunk"
    ),
    top_genes: Optional[int] = typer.Option(
        2000,
        "--top-genes",
        help="Number of top expressed genes per cell. If not specified, defaults to 2000. Pass 0 or a very large number to use all genes (or modify code to accept None)."
    ),
    compression: str = typer.Option(
        "zstd",
        "--compression",
        help="Compression algorithm for parquet files. Options: uncompressed, snappy, gzip, lzo, brotli, lz4, zstd"
    ),
    compression_level: int = typer.Option(
        3,
        "--compression-level",
        help="Compression level (1-9 for zstd/gzip, 1-11 for brotli)"
    ),
    use_pyarrow: bool = typer.Option(
        True,
        "--use-pyarrow/--no-pyarrow",
        help="Use pyarrow backend for parquet writes (faster)"
    ),
    test_size: float = typer.Option(
        0.05,
        "--test-size",
        help="Proportion of data for test set"
    ),
    skip_train_test_split: bool = typer.Option(
        False,
        "--skip-train-test-split",
        help="Skip train/test split and produce single parquet dataset (default: False)"
    ),
    log_dir: Optional[Path] = typer.Option(
        Path("./logs"),
        "--log-dir",
        help="Directory for log files (separate log per file)"
    ),
    log_stdout: bool = typer.Option(
        True,
        "--log-stdout/--no-log-stdout",
        help="Mirror Eliot logs to stdout"
    ),
    batch_mode: bool = typer.Option(
        False,
        "--batch-mode",
        help="Process all h5ad files in input directory (default: False, process single file)"
    ),
    skip_existing: bool = typer.Option(
        False,
        "--skip-existing",
        help="Skip datasets that already have output files (default: False)"
    ),
    lookup_publication: bool = typer.Option(
        True,
        "--lookup-publication/--no-lookup-publication",
        help="Enable/disable CellxGene API lookup for publication metadata. By default, auto-detects cellxgene datasets and joins if found. Use --no-lookup-publication to disable."
    ),
    filter_by_age: bool = typer.Option(
        True,
        "--filter-age/--no-age-filter",
        help="Filter out cells with null age values. Default: True (filters out cells without age). Use --no-age-filter to keep all cells regardless of age."
    ),
    gene_lists_dir: Optional[Path] = typer.Option(
        Path("./data/shared/gene_lists"),
        "--gene-lists-dir",
        "-g",
        help="Directory containing gene list .txt files (one gene symbol per row). Creates both full_gene_sentence (top 2K genes) and cell_sentence (filtered to genes in lists) columns. Default: ./data/shared/gene_lists. Use empty string or non-existent path to disable."
    ),
    max_file_size_mb: Optional[float] = typer.Option(
        None,
        "--max-file-size-mb",
        help="Maximum file size in MB to process (e.g., 12000 for 12 GB). Files larger than this will be skipped."
    ),
) -> None:
    """Run the preprocessing pipeline.
    
    One-step streaming approach that processes each h5ad file in a single pass:
    1. One-step conversion per file:
       - Read h5ad chunks
       - Create cell sentences
       - Extract age from development_stage
       - Split into train/test (if not skipped)
       - Write directly to output
    2. Upload to HuggingFace (per file, if repo_id and token provided)
    
    Memory Management:
    - No interim files are created (everything happens in one streaming pass)
    - Garbage collection is forced after each file
    - This ensures efficient memory usage when processing terabytes of data
    
    Batch Mode:
    - Use --batch-mode to process all h5ad files in a directory
    - Each file gets its own output directory and log file
    - Failures are logged but don't stop processing of other files
    - If token is provided, uploads happen after each successful file processing
    
    If --skip-train-test-split is used, the data will remain in a single parquet directory,
    allowing users on HuggingFace to decide on their own splitting strategy.
    """
    # Setup logging once at the start (not per-file)
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
        if batch_mode:
            # In batch mode, use a global log file
            global_log = log_dir / "batch_pipeline.log"
        else:
            # In single file mode, use a simple pipeline log
            global_log = log_dir / "pipeline.log"
        json_path = global_log.with_suffix('.json')
        to_nice_file(output_file=json_path, rendered_file=global_log)
        to_nice_stdout(output_file=json_path)
        typer.echo(f"Logging to: {global_log}")
    if log_stdout:
        to_file(sys.stdout)
    
    with start_action(action_type="cli_run", batch_mode=batch_mode) as action:
        # Validate output directory - prevent writing to data/test (reserved for code tests)
        output_dir_resolved = output_dir.resolve()
        test_dir_resolved = Path("./data/test").resolve()
        if output_dir_resolved == test_dir_resolved:
            typer.secho(
                f"Error: Cannot use 'data/test' as output directory. "
                f"This directory is reserved for code tests. Use 'data/output' instead.",
                fg=typer.colors.RED
            )
            typer.echo(f"  Provided: {output_dir}")
            typer.echo(f"  Use: --output-dir ./data/output (or omit for default)")
            raise typer.Exit(1)
        
        # Determine which files to process
        h5ad_files: list[Path] = []
        
        if h5ad_path is not None and h5ad_path.exists():
            if h5ad_path.is_dir():
                # Directory provided, process all h5ad files in it
                h5ad_files = sorted(list(h5ad_path.glob("*.h5ad")))
                typer.echo(f"Found {len(h5ad_files)} h5ad file(s) in {h5ad_path}")
                action.log(message_type="files_found_in_directory", count=len(h5ad_files), directory=str(h5ad_path))
            elif h5ad_path.suffix == ".h5ad":
                # Single file provided
                h5ad_files = [h5ad_path]
                action.log(message_type="single_file_provided", file_path=str(h5ad_path))
            else:
                typer.secho(f"Error: {h5ad_path} is not a valid h5ad file or directory", fg=typer.colors.RED)
                action.log(message_type="invalid_path_error", path=str(h5ad_path))
                raise typer.Exit(1)
        else:
            # Auto-detect from input_dir
            h5ad_files = sorted(list(input_dir.glob("*.h5ad")))
            if h5ad_files:
                if batch_mode or len(h5ad_files) > 1:
                    typer.echo(f"Auto-detected {len(h5ad_files)} h5ad file(s) in {input_dir}")
                    action.log(message_type="files_auto_detected", count=len(h5ad_files), input_dir=str(input_dir), batch_mode=batch_mode)
                else:
                    # Single file, use only first one
                    h5ad_files = [h5ad_files[0]]
                    typer.echo(f"Auto-detected h5ad file: {h5ad_files[0]}")
                    action.log(message_type="single_file_auto_detected", file_path=str(h5ad_files[0]), input_dir=str(input_dir))
            else:
                typer.secho(f"Error: Could not find h5ad file. Searched in {input_dir}", fg=typer.colors.RED)
                typer.echo("Please provide h5ad_path or download a dataset first using 'preprocess download'")
                action.log(message_type="no_files_found_error", input_dir=str(input_dir))
                raise typer.Exit(1)
        
        # Decide batch vs single mode
        process_multiple = batch_mode or len(h5ad_files) > 1
        
        # Process files
        results: list[tuple[str, bool, str, float, Path]] = []
        skipped_datasets: list[tuple[str, Path]] = []
        
        for idx, h5ad_file in enumerate(h5ad_files, 1):
            dataset_name = sanitize_dataset_name(h5ad_file.stem)
            dataset_output_path = output_dir / dataset_name
            
            if process_multiple:
                # Progress indicator for batch mode
                typer.echo(f"\n[{idx}/{len(h5ad_files)}] Processing: {dataset_name}")
            
            # Check file size if limit is specified
            if max_file_size_mb is not None:
                file_size_mb = h5ad_file.stat().st_size / (1024 * 1024)
                if file_size_mb > max_file_size_mb:
                    action.log(
                        message_type="skipping_large_file",
                        dataset_name=dataset_name,
                        file=h5ad_file.name,
                        file_size_mb=round(file_size_mb, 2),
                        max_file_size_mb=max_file_size_mb
                    )
                    skip_message = f"skipped (file too large: {file_size_mb:.2f} MB > {max_file_size_mb} MB)"
                    typer.echo(f"  Skipping (file size {file_size_mb:.2f} MB exceeds limit {max_file_size_mb} MB)")
                    results.append((dataset_name, False, skip_message, 0.0, dataset_output_path))
                    continue
            
            # Check if output already exists and skip if flag is enabled
            if skip_existing and check_output_exists(output_dir, dataset_name, skip_train_test_split):
                action.log(message_type="dataset_skipped", dataset_name=dataset_name, reason="output_already_exists", output_path=str(dataset_output_path))
                typer.echo("  Skipping (output already exists)")
                skipped_datasets.append((dataset_name, dataset_output_path))
                continue
            
            # Process the file
            success, message, processing_time, dataset_output_path = _process_single_file(
                h5ad_path=h5ad_file,
                output_dir=output_dir,
                chunk_size=chunk_size,
                top_genes=top_genes,
                compression=compression,
                compression_level=compression_level,
                use_pyarrow=use_pyarrow,
                test_size=test_size,
                skip_train_test_split=skip_train_test_split,
                repo_id=repo_id,
                token=token,
                join_collection=lookup_publication,
                filter_by_age=filter_by_age,
                gene_lists_dir=gene_lists_dir,
            )
            
            results.append((dataset_name, success, message, processing_time, dataset_output_path))
        
        # Summary
        successful = sum(1 for _, success, _, _, _ in results if success)
        failed = len(results) - successful
        total_processed = len(results) + len(skipped_datasets)
        
        typer.echo(f"\nProcessing complete: {successful}/{total_processed} successful")
        if failed > 0:
            typer.echo(f"  Failed: {failed}")
        if skipped_datasets:
            typer.echo(f"  Skipped: {len(skipped_datasets)}")
        
        action.log(
            message_type="pipeline_summary",
            total_files=total_processed,
            successful=successful,
            failed=failed,
            skipped=len(skipped_datasets),
            input_dir=str(input_dir),
            output_dir=str(output_dir),
            log_dir=str(log_dir) if log_dir else None
        )
        
        # Show failed datasets if any
        if failed > 0:
            typer.echo("\nFailed datasets:")
            for dataset_name, success, message, _, _ in results:
                if not success:
                    typer.echo(f"  ✗ {dataset_name}: {message}")
        
        typer.echo(f"\nOutput: {output_dir}")
        if log_dir:
            typer.echo(f"Logs: {log_dir}")
        
        # Write summary TSV file if processing multiple files
        if process_multiple:
            summary_file = output_dir / "batch_processing_summary.tsv"
            
            with start_action(action_type="write_batch_summary", summary_file=str(summary_file)) as action:
                import polars as pl
                
                # Prepare data for TSV
                summary_data = []
                
                # Add skipped datasets
                for dataset_name, dataset_output_path in skipped_datasets:
                    summary_data.append({
                        'dataset_name': dataset_name,
                        'status': 'SKIPPED',
                        'processing_time_seconds': 0.0,
                        'processing_time_formatted': '00:00:00',
                        'output_path': str(dataset_output_path),
                        'message': 'Output files already exist'
                    })
                
                # Add processed datasets
                for dataset_name, success, message, processing_time, dataset_output_path in results:
                    status = "SUCCESS" if success else "FAILED"
                    
                    # Format time as HH:MM:SS
                    hours = int(processing_time // 3600)
                    minutes = int((processing_time % 3600) // 60)
                    seconds = int(processing_time % 60)
                    time_formatted = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
                    
                    # For failed datasets, mark crash in message
                    if not success:
                        message = f"CRASH: {message}"
                    
                    summary_data.append({
                        'dataset_name': dataset_name,
                        'status': status,
                        'processing_time_seconds': round(processing_time, 2),
                        'processing_time_formatted': time_formatted,
                        'output_path': str(dataset_output_path),
                        'message': message
                    })
                
                # Create DataFrame and write to TSV
                df = pl.DataFrame(summary_data)
                df.write_csv(summary_file, separator='\t')
                
                action.log(
                    message_type="batch_summary_written",
                    summary_file=str(summary_file),
                    total_datasets=total_processed,
                    successful=successful,
                    failed=failed,
                    skipped=len(skipped_datasets)
                )
                
                typer.echo(f"Summary: {summary_file}")


def main() -> None:
    """Main entrypoint that shows help by default."""
    import sys
    # If no arguments provided, show help
    if len(sys.argv) == 1:
        sys.argv.append("--help")
    app()


if __name__ == "__main__":
    main()

