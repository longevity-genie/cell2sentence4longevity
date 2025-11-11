"""Preprocessing command-line interface for cell2sentence pipeline."""

from pathlib import Path
from typing import Optional
import gc
import shutil
import re

import typer
from eliot import start_action
from pycomfort.logging import to_nice_file
from dotenv import load_dotenv

from cell2sentence4longevity.preprocessing import (
    create_hgnc_mapper,
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
) -> None:
    """Download dataset from a URL.
    
    Downloads datasets (typically h5ad files) to the input directory.
    By default, skips download if file already exists. Use --force to re-download.
    """
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        json_path = log_file.with_suffix('.json')
        to_nice_file(output_file=json_path, rendered_file=log_file)
    
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
def hgnc_mapper(
    interim_dir: Path = typer.Option(
        Path("./data/interim"),
        "--interim-dir",
        "-i",
        help="Directory to save interim files (HGNC mappers)"
    ),
    log_file: Optional[Path] = typer.Option(
        None,
        "--log-file",
        "-l",
        help="Path to eliot log file"
    ),
) -> None:
    """Create HGNC gene mapper.
    
    Downloads official gene mappings from HGNC to convert Ensembl IDs to gene symbols.
    """
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        json_path = log_file.with_suffix('.json')
        to_nice_file(output_file=json_path, rendered_file=log_file)
    
    with start_action(action_type="cli_hgnc_mapper") as action:
        typer.echo("Creating HGNC mapper...")
        action.log(message_type="hgnc_mapper_creation_started", interim_dir=str(interim_dir))
        create_hgnc_mapper(interim_dir)
        mapper_path = interim_dir / 'hgnc_mappers.pkl'
        typer.secho("✓ HGNC mapper created successfully", fg=typer.colors.GREEN)
        typer.echo(f"Output: {mapper_path}")
        action.log(message_type="hgnc_mapper_creation_completed", output_path=str(mapper_path))


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
) -> None:
    """Upload to HuggingFace.
    
    Uploads the processed data to HuggingFace in a single batch commit.
    """
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        json_path = log_file.with_suffix('.json')
        to_nice_file(output_file=json_path, rendered_file=log_file)
    
    with start_action(action_type="cli_upload") as action:
        typer.echo("Uploading to HuggingFace...")
        action.log(message_type="upload_started", repo_id=repo_id, output_dir=str(output_dir))
        upload_to_huggingface(
            data_splits_dir=output_dir,
            token=token,
            repo_id=repo_id,
            readme_path=readme_path
        )
        dataset_url = f"https://huggingface.co/datasets/{repo_id}"
        typer.secho("✓ Upload completed successfully", fg=typer.colors.GREEN)
        typer.echo(f"Dataset: {dataset_url}")
        action.log(message_type="upload_completed", repo_id=repo_id, dataset_url=dataset_url)


def _process_single_file(
    h5ad_path: Path,
    interim_dir: Path,
    output_dir: Path,
    chunk_size: int,
    top_genes: int,
    compression: str,
    compression_level: int,
    use_pyarrow: bool,
    test_size: float,
    skip_train_test_split: bool,
    repo_id: Optional[str],
    token: Optional[str],
    mappers_path: Path | None,
    keep_interim: bool = False,
    join_collection: bool = True,
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
        interim_dir: Base interim directory (unused in one-step approach)
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
        mappers_path: Path to HGNC mappers
        keep_interim: Whether to keep interim files (unused in one-step approach)
        
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
            typer.echo("="*80)
            typer.echo(f"Processing: {dataset_name}")
            typer.echo("Converting h5ad, extracting age, and creating train/test split in one pass")
            typer.echo("="*80)
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
                mappers_path=mappers_path,
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
                join_collection=join_collection
            )
            
            typer.secho(f"✓ Conversion complete for {dataset_name}\n", fg=typer.colors.GREEN)
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
                typer.echo("="*80)
                typer.echo(f"Uploading: {dataset_name} to HuggingFace")
                typer.echo("="*80)
                # Upload to same repository as subfolder (dataset_name creates subfolder in repo)
                action.log(message_type="upload_started", dataset_name=dataset_name, repo_id=repo_id, upload_dir=str(upload_dir))
                upload_to_huggingface(
                    data_splits_dir=upload_dir,
                    token=token,
                    repo_id=repo_id,
                    dataset_name=dataset_name
                )
                dataset_url = f"https://huggingface.co/datasets/{repo_id}"
                typer.secho(f"✓ Upload complete for {dataset_name}\n", fg=typer.colors.GREEN)
                typer.echo(f"Dataset: {dataset_url} (subfolder: {dataset_name})")
                action.log(message_type="upload_completed", dataset_name=dataset_name, repo_id=repo_id, dataset_url=dataset_url)
            
            # Final garbage collection
            gc.collect()
            
            processing_time = time.time() - start_time
            action.log(message_type="file_processing_success", dataset_name=dataset_name, processing_time_seconds=round(processing_time, 2))
            return True, f"Successfully processed {dataset_name}", processing_time, dataset_output_dir
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Failed to process {dataset_name}: {str(e)}"
            action.log(message_type="file_processing_error", dataset_name=dataset_name, error=str(e), processing_time_seconds=round(processing_time, 2))
            typer.secho(f"✗ Error processing {dataset_name}: {e}", fg=typer.colors.RED)
            
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
    interim_dir: Path = typer.Option(
        Path("./data/interim"),
        "--interim-dir",
        help="Directory for interim files"
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
    top_genes: int = typer.Option(
        2000,
        "--top-genes",
        help="Number of top expressed genes per cell"
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
    batch_mode: bool = typer.Option(
        False,
        "--batch-mode",
        help="Process all h5ad files in input directory (default: False, process single file)"
    ),
    keep_interim: bool = typer.Option(
        False,
        "--keep-interim",
        help="Keep interim parquet files after processing (default: False, clean up to save space)"
    ),
    mappers_path: Path | None = typer.Option(
        None,
        "--mappers",
        "-m",
        help="Path to HGNC mappers pickle file (optional, only created/used if needed or explicitly provided)"
    ),
    create_hgnc: bool = typer.Option(
        False,
        "--create-hgnc",
        help="Force creation of HGNC mapper (default: False, only created if needed)"
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
) -> None:
    """Run the preprocessing pipeline.
    
    One-step streaming approach that processes each h5ad file in a single pass:
    1. Create HGNC mapper (optional, only if --create-hgnc or if needed)
    2. One-step conversion per file:
       - Read h5ad chunks
       - Create cell sentences
       - Extract age from development_stage
       - Split into train/test (if not skipped)
       - Write directly to output
    3. Upload to HuggingFace (per file, if repo_id and token provided)
    
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
    with start_action(action_type="cli_run", batch_mode=batch_mode, keep_interim=keep_interim) as action:
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
        
        # Create HGNC mapper (optional, only if requested or if mappers_path not provided)
        mappers_path_final = mappers_path
        if create_hgnc:
            typer.echo("\n" + "="*80)
            typer.echo("Creating HGNC mapper")
            typer.echo("="*80)
            action.log(message_type="hgnc_mapper_creation_requested", interim_dir=str(interim_dir))
            try:
                create_hgnc_mapper(interim_dir)
                typer.secho("✓ HGNC mapper created\n", fg=typer.colors.GREEN)
                mappers_path_final = interim_dir / "hgnc_mappers.pkl"
                action.log(message_type="hgnc_mapper_created", mappers_path=str(mappers_path_final))
            except Exception as e:
                typer.secho(f"⚠ Warning: Failed to create HGNC mapper: {e}", fg=typer.colors.YELLOW)
                typer.echo("Will proceed without HGNC mapper (will use gene symbols from h5ad if available)\n")
                action.log(message_type="hgnc_mapper_creation_failed", error=str(e))
                mappers_path_final = None
        elif mappers_path is None:
            typer.echo("\n" + "="*80)
            typer.echo("Skipping HGNC mapper creation (use --create-hgnc to create)")
            typer.echo("="*80)
            typer.echo("HGNC will only be used if needed (AnnData has Ensembl IDs without gene symbols)\n")
            action.log(message_type="hgnc_mapper_skipped", reason="not_requested_and_no_path_provided")
            mappers_path_final = None
        
        # Process files
        results: list[tuple[str, bool, str, float, Path]] = []
        skipped_datasets: list[tuple[str, Path]] = []
        
        for idx, h5ad_file in enumerate(h5ad_files, 1):
            dataset_name = sanitize_dataset_name(h5ad_file.stem)
            dataset_output_path = output_dir / dataset_name
            
            typer.echo("\n" + "="*80)
            if process_multiple:
                typer.echo(f"Processing file {idx}/{len(h5ad_files)}: {dataset_name}")
            else:
                typer.echo(f"Processing: {dataset_name}")
            typer.echo("="*80)
            
            # Check if output already exists and skip if flag is enabled
            if skip_existing and check_output_exists(output_dir, dataset_name, skip_train_test_split):
                typer.secho(f"⏭ Skipping {dataset_name}: output files already exist", fg=typer.colors.YELLOW)
                typer.echo(f"  Output directory: {dataset_output_path}")
                action.log(message_type="dataset_skipped", dataset_name=dataset_name, reason="output_already_exists", output_path=str(dataset_output_path))
                skipped_datasets.append((dataset_name, dataset_output_path))
                continue
            
            # Setup per-file logging
            if log_dir:
                file_log = log_dir / dataset_name / "pipeline.log"
                file_log.parent.mkdir(parents=True, exist_ok=True)
                json_path = file_log.with_suffix('.json')
                to_nice_file(output_file=json_path, rendered_file=file_log)
            
            # Process the file
            success, message, processing_time, dataset_output_path = _process_single_file(
                h5ad_path=h5ad_file,
                interim_dir=interim_dir,
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
                mappers_path=mappers_path_final,
                keep_interim=keep_interim,
                join_collection=lookup_publication,
            )
            
            results.append((dataset_name, success, message, processing_time, dataset_output_path))
        
        # Summary
        typer.echo("\n" + "="*80)
        typer.secho("Pipeline Complete - Summary", fg=typer.colors.GREEN, bold=True)
        typer.echo("="*80)
        
        successful = sum(1 for _, success, _, _, _ in results if success)
        failed = len(results) - successful
        total_processed = len(results) + len(skipped_datasets)
        
        typer.echo(f"Total files: {total_processed}")
        typer.echo(f"Successful: {successful}")
        if failed > 0:
            typer.echo(f"Failed: {failed}")
        if skipped_datasets:
            typer.echo(f"Skipped: {len(skipped_datasets)}")
        
        action.log(
            message_type="pipeline_summary",
            total_files=total_processed,
            successful=successful,
            failed=failed,
            skipped=len(skipped_datasets),
            input_dir=str(input_dir),
            interim_dir=str(interim_dir),
            output_dir=str(output_dir),
            log_dir=str(log_dir) if log_dir else None
        )
        
        typer.echo("\nDetails:")
        # Show skipped datasets first
        for dataset_name, dataset_output_path in skipped_datasets:
            typer.secho(f"  ⏭ {dataset_name}: Skipped (output already exists)", fg=typer.colors.YELLOW)
            typer.echo(f"    Output: {dataset_output_path}")
        # Show processed datasets
        for dataset_name, success, message, processing_time, _ in results:
            status = "✓" if success else "✗"
            color = typer.colors.GREEN if success else typer.colors.RED
            typer.secho(f"  {status} {dataset_name}: {message}", fg=color)
        
        typer.echo(f"\nOutput directories:")
        typer.echo(f"  Input: {input_dir}")
        typer.echo(f"  Interim: {interim_dir}")
        typer.echo(f"  Output: {output_dir}")
        if log_dir:
            typer.echo(f"  Logs: {log_dir}")
        
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
                
                typer.echo(f"\n✓ Batch processing summary written to: {summary_file}")
                typer.echo(f"  Summary contains timing and output paths for all processed datasets")


def main() -> None:
    """Main entrypoint that shows help by default."""
    import sys
    # If no arguments provided, show help
    if len(sys.argv) == 1:
        sys.argv.append("--help")
    app()


if __name__ == "__main__":
    main()

