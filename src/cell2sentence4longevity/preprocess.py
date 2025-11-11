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
    convert_h5ad_to_parquet,
    add_age_and_cleanup,
    create_train_test_split,
    upload_to_huggingface,
    download_dataset,
)

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
    
    with start_action(action_type="cli_download"):
        if force:
            typer.echo(f"Force downloading dataset from {url}...")
        else:
            typer.echo(f"Downloading dataset from {url} (skipping if already exists)...")
        output_path = download_dataset(url, input_dir, filename, force=force)
        typer.secho("✓ Download completed successfully", fg=typer.colors.GREEN)
        typer.echo(f"Saved to: {output_path}")


@app.command()
def step1_hgnc_mapper(
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
    """Step 1: Create HGNC gene mapper.
    
    Downloads official gene mappings from HGNC to convert Ensembl IDs to gene symbols.
    """
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        json_path = log_file.with_suffix('.json')
        to_nice_file(output_file=json_path, rendered_file=log_file)
    
    with start_action(action_type="cli_step1_hgnc_mapper"):
        typer.echo("Step 1: Creating HGNC mapper...")
        create_hgnc_mapper(interim_dir)
        typer.secho("✓ HGNC mapper created successfully", fg=typer.colors.GREEN)
        typer.echo(f"Output: {interim_dir / 'hgnc_mappers.pkl'}")


@app.command()
def step2_convert_h5ad(
    h5ad_path: Path | None = typer.Argument(
        None,
        help="Path to AIDA h5ad file or directory containing h5ad files (optional, auto-detects from --input-dir if not provided)"
    ),
    input_dir: Path = typer.Option(
        Path("./data/input"),
        "--input-dir",
        help="Directory containing input h5ad files (auto-detects if h5ad_path not provided)"
    ),
    mappers_path: Path | None = typer.Option(
        None,
        "--mappers",
        "-m",
        help="Path to HGNC mappers pickle file (optional, only used if needed or explicitly provided)"
    ),
    interim_dir: Path = typer.Option(
        Path("./data/interim/parquet_chunks"),
        "--interim-dir",
        "-i",
        help="Directory to save interim parquet chunks"
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
        "-t",
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
    log_dir: Optional[Path] = typer.Option(
        None,
        "--log-dir",
        "-l",
        help="Directory for log files (separate log per file in batch mode)"
    ),
    batch_mode: bool = typer.Option(
        False,
        "--batch-mode",
        help="Process all h5ad files in input directory (default: False, process single file)"
    ),
) -> None:
    """Step 2: Convert h5ad to parquet with cell sentences.
    
    Transforms the AIDA h5ad file into parquet chunks, creating "cell sentences" -
    space-separated gene symbols ordered by expression level.
    
    Batch Mode:
    - Use --batch-mode to process all h5ad files in a directory
    - Each file gets its own output subdirectory and optional log file
    - Failures are logged but don't stop processing of other files
    """
    with start_action(action_type="cli_step2_convert_h5ad", batch_mode=batch_mode):
        # Determine which files to process
        h5ad_files: list[Path] = []
        
        if h5ad_path is not None and h5ad_path.exists():
            if h5ad_path.is_dir():
                # Directory provided, process all h5ad files in it
                h5ad_files = sorted(list(h5ad_path.glob("*.h5ad")))
                typer.echo(f"Found {len(h5ad_files)} h5ad file(s) in {h5ad_path}")
            elif h5ad_path.suffix == ".h5ad":
                # Single file provided
                h5ad_files = [h5ad_path]
            else:
                typer.secho(f"Error: {h5ad_path} is not a valid h5ad file or directory", fg=typer.colors.RED)
                raise typer.Exit(1)
        else:
            # Auto-detect from input_dir
            h5ad_files = sorted(list(input_dir.glob("*.h5ad")))
            if h5ad_files:
                if batch_mode or len(h5ad_files) > 1:
                    typer.echo(f"Auto-detected {len(h5ad_files)} h5ad file(s) in {input_dir}")
                else:
                    # Single file, use only first one
                    h5ad_files = [h5ad_files[0]]
                    typer.echo(f"Auto-detected h5ad file: {h5ad_files[0]}")
            else:
                typer.secho(f"Error: Could not find h5ad file. Searched in {input_dir}", fg=typer.colors.RED)
                typer.echo("Please provide h5ad_path or download a dataset first using 'preprocess download'")
                raise typer.Exit(1)
        
        # Decide batch vs single mode
        process_multiple = batch_mode or len(h5ad_files) > 1
        
        # Process files
        results: list[tuple[str, bool, str]] = []
        
        for idx, h5ad_file in enumerate(h5ad_files, 1):
            dataset_name = sanitize_dataset_name(h5ad_file.stem)
            
            if process_multiple:
                typer.echo(f"\nProcessing file {idx}/{len(h5ad_files)}: {dataset_name}")
            else:
                typer.echo(f"Step 2: Converting {dataset_name} to parquet...")
            
            # Setup per-file logging
            if log_dir and process_multiple:
                file_log = log_dir / dataset_name / "convert_h5ad.log"
                file_log.parent.mkdir(parents=True, exist_ok=True)
                json_path = file_log.with_suffix('.json')
                to_nice_file(output_file=json_path, rendered_file=file_log)
            
            try:
                # Each file gets its own subdirectory
                file_output_dir = interim_dir / dataset_name if process_multiple else interim_dir
                
                convert_h5ad_to_parquet(
                    h5ad_file, mappers_path, file_output_dir, chunk_size, top_genes,
                    compression=compression, compression_level=compression_level, 
                    use_pyarrow=use_pyarrow, dataset_name=dataset_name
                )
                
                typer.secho(f"✓ Conversion completed for {dataset_name}", fg=typer.colors.GREEN)
                typer.echo(f"Output: {file_output_dir}")
                results.append((dataset_name, True, "Success"))
                
            except Exception as e:
                error_msg = f"Failed to convert {dataset_name}: {str(e)}"
                typer.secho(f"✗ Error: {error_msg}", fg=typer.colors.RED)
                results.append((dataset_name, False, error_msg))
                
                # Continue processing other files in batch mode
                if not process_multiple:
                    raise
        
        # Summary for batch mode
        if process_multiple:
            typer.echo("\n" + "="*80)
            typer.secho("CONVERSION SUMMARY", fg=typer.colors.GREEN, bold=True)
            typer.echo("="*80)
            
            successful = sum(1 for _, success, _ in results if success)
            failed = len(results) - successful
            
            typer.echo(f"Total files: {len(results)}")
            typer.echo(f"Successful: {successful}")
            if failed > 0:
                typer.echo(f"Failed: {failed}")
            
            typer.echo("\nDetails:")
            for dataset_name, success, message in results:
                status = "✓" if success else "✗"
                color = typer.colors.GREEN if success else typer.colors.RED
                typer.secho(f"  {status} {dataset_name}: {message}", fg=color)


@app.command()
def step3_add_age(
    interim_dir: Path = typer.Option(
        Path("./data/interim/parquet_chunks"),
        "--interim-dir",
        "-i",
        help="Directory containing interim parquet chunks (can contain subdirectories for multiple datasets)"
    ),
    log_file: Optional[Path] = typer.Option(
        None,
        "--log-file",
        "-l",
        help="Path to eliot log file"
    ),
    batch_mode: bool = typer.Option(
        False,
        "--batch-mode",
        help="Process all dataset subdirectories in interim_dir (default: False)"
    ),
) -> None:
    """Step 3: Add age column and cleanup.
    
    Extracts age as integer from development_stage field and ensures proper column naming.
    
    Batch Mode:
    - Use --batch-mode to process all dataset subdirectories
    - Each subdirectory is treated as a separate dataset
    """
    if log_file and not batch_mode:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        json_path = log_file.with_suffix('.json')
        to_nice_file(output_file=json_path, rendered_file=log_file)
    
    with start_action(action_type="cli_step3_add_age", batch_mode=batch_mode):
        if batch_mode:
            # Process all subdirectories
            dataset_dirs = [d for d in interim_dir.iterdir() if d.is_dir()]
            typer.echo(f"Found {len(dataset_dirs)} dataset(s) in {interim_dir}")
            
            for dataset_dir in dataset_dirs:
                dataset_name = dataset_dir.name
                typer.echo(f"\nProcessing: {dataset_name}")
                try:
                    add_age_and_cleanup(dataset_dir)
                    typer.secho(f"✓ Age added and cleanup completed for {dataset_name}", fg=typer.colors.GREEN)
                except Exception as e:
                    typer.secho(f"✗ Error processing {dataset_name}: {e}", fg=typer.colors.RED)
        else:
            # Single directory processing
            typer.echo("Step 3: Adding age and cleaning up...")
            add_age_and_cleanup(interim_dir)
            typer.secho("✓ Age added and cleanup completed", fg=typer.colors.GREEN)


@app.command()
def step4_train_test_split(
    interim_dir: Path = typer.Option(
        Path("./data/interim/parquet_chunks"),
        "--interim-dir",
        "-i",
        help="Directory containing interim parquet chunks"
    ),
    output_dir: Path = typer.Option(
        Path("./data/output"),
        "--output-dir",
        "-o",
        help="Directory to save final train/test splits"
    ),
    test_size: float = typer.Option(
        0.05,
        "--test-size",
        "-t",
        help="Proportion of data for test set"
    ),
    random_state: int = typer.Option(
        42,
        "--random-state",
        "-r",
        help="Random seed for reproducibility"
    ),
    chunk_size: int = typer.Option(
        10000,
        "--chunk-size",
        "-c",
        help="Number of cells per output chunk"
    ),
    log_file: Optional[Path] = typer.Option(
        None,
        "--log-file",
        "-l",
        help="Path to eliot log file"
    ),
) -> None:
    """Step 4: Create stratified train/test split.
    
    Creates a train/test split stratified by age to maintain age distribution in both sets.
    """
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        json_path = log_file.with_suffix('.json')
        to_nice_file(output_file=json_path, rendered_file=log_file)
    
    with start_action(action_type="cli_step4_train_test_split"):
        typer.echo("Step 4: Creating train/test split...")
        create_train_test_split(interim_dir, output_dir, test_size, random_state, chunk_size)
        typer.secho("✓ Train/test split created successfully", fg=typer.colors.GREEN)
        typer.echo(f"Output: {output_dir}")


@app.command()
def step5_upload(
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
    """Step 5: Upload to HuggingFace.
    
    Uploads the processed data to HuggingFace in a single batch commit.
    """
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        json_path = log_file.with_suffix('.json')
        to_nice_file(output_file=json_path, rendered_file=log_file)
    
    with start_action(action_type="cli_step5_upload"):
        typer.echo("Step 5: Uploading to HuggingFace...")
        upload_to_huggingface(output_dir, token, repo_id, readme_path)
        typer.secho("✓ Upload completed successfully", fg=typer.colors.GREEN)
        typer.echo(f"Dataset: https://huggingface.co/datasets/{repo_id}")


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
) -> tuple[bool, str]:
    """Process a single h5ad file through the entire pipeline.
    
    Args:
        h5ad_path: Path to the h5ad file
        interim_dir: Base interim directory
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
        keep_interim: Whether to keep interim files after processing
        
    Returns:
        Tuple of (success: bool, message: str)
    """
    dataset_name = sanitize_dataset_name(h5ad_path.stem)
    parquet_dir = interim_dir / "parquet_chunks" / dataset_name
    
    with start_action(action_type="process_single_file", dataset_name=dataset_name, h5ad_path=str(h5ad_path)) as action:
        try:
            # Step 2: Convert h5ad to parquet
            typer.echo("="*80)
            typer.echo(f"STEP 2: Converting {dataset_name} to parquet")
            typer.echo("="*80)
            convert_h5ad_to_parquet(
                h5ad_path, mappers_path, parquet_dir, chunk_size, top_genes,
                compression=compression, compression_level=compression_level, 
                use_pyarrow=use_pyarrow, dataset_name=dataset_name
            )
            typer.secho(f"✓ Step 2 complete for {dataset_name}\n", fg=typer.colors.GREEN)
            
            # Force garbage collection to free memory
            gc.collect()
            
            # Step 3: Add age and cleanup
            typer.echo("="*80)
            typer.echo(f"STEP 3: Adding age and cleaning up {dataset_name}")
            typer.echo("="*80)
            add_age_and_cleanup(parquet_dir)
            typer.secho(f"✓ Step 3 complete for {dataset_name}\n", fg=typer.colors.GREEN)
            
            # Force garbage collection
            gc.collect()
            
            # Step 4 (optional, based on skip_train_test_split flag)
            if skip_train_test_split:
                typer.echo(f"\n⚠ Skipping Step 4 (train/test split) for {dataset_name}")
                upload_dir = parquet_dir
            else:
                typer.echo("="*80)
                typer.echo(f"STEP 4: Creating train/test split for {dataset_name}")
                typer.echo("="*80)
                dataset_output_dir = output_dir / dataset_name
                create_train_test_split(parquet_dir, dataset_output_dir, test_size, 42, chunk_size)
                typer.secho(f"✓ Step 4 complete for {dataset_name}\n", fg=typer.colors.GREEN)
                upload_dir = dataset_output_dir
                
                # Clean up interim files if not keeping them
                if not keep_interim:
                    typer.echo(f"Cleaning up interim files for {dataset_name}...")
                    if parquet_dir.exists():
                        shutil.rmtree(parquet_dir)
                        action.log(message_type="interim_cleanup", dataset_name=dataset_name, cleaned_dir=str(parquet_dir))
                        typer.secho(f"✓ Interim files cleaned up", fg=typer.colors.GREEN)
            
            # Force garbage collection
            gc.collect()
            
            # Step 5 (optional)
            if repo_id and token:
                typer.echo("="*80)
                typer.echo(f"STEP 5: Uploading {dataset_name} to HuggingFace")
                typer.echo("="*80)
                # Append dataset name to repo_id for multi-file uploads
                dataset_repo_id = f"{repo_id}-{dataset_name}" if repo_id else repo_id
                upload_to_huggingface(upload_dir, token, dataset_repo_id, None)
                typer.secho(f"✓ Step 5 complete for {dataset_name}\n", fg=typer.colors.GREEN)
                typer.echo(f"Dataset: https://huggingface.co/datasets/{dataset_repo_id}")
            
            # Final garbage collection
            gc.collect()
            
            action.log(message_type="file_processing_success", dataset_name=dataset_name, interim_kept=keep_interim)
            return True, f"Successfully processed {dataset_name}"
            
        except Exception as e:
            error_msg = f"Failed to process {dataset_name}: {str(e)}"
            action.log(message_type="file_processing_error", dataset_name=dataset_name, error=str(e))
            typer.secho(f"✗ Error processing {dataset_name}: {e}", fg=typer.colors.RED)
            
            # Try to clean up partial interim files on error (unless keep_interim is True)
            if not keep_interim and parquet_dir.exists():
                try:
                    typer.echo(f"Cleaning up partial interim files for {dataset_name}...")
                    shutil.rmtree(parquet_dir)
                    action.log(message_type="partial_interim_cleanup", dataset_name=dataset_name, reason="error")
                except Exception as cleanup_error:
                    action.log(message_type="cleanup_failed", error=str(cleanup_error))
            
            # Force garbage collection even on error
            gc.collect()
            
            return False, error_msg


@app.command()
def run_all(
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
        None,
        "--repo-id",
        "-r",
        help="HuggingFace repository ID (e.g., 'username/dataset-name'). For multiple files, dataset name will be appended."
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
) -> None:
    """Run all pipeline steps sequentially.
    
    This command runs all preprocessing steps in order:
    1. Create HGNC mapper (optional, only if --create-hgnc or if needed)
    2. Convert h5ad to parquet (per file)
    3. Add age and cleanup (per file)
    4. Create train/test split (per file, optional, can be skipped with --skip-train-test-split)
    5. Upload to HuggingFace (per file, if repo_id and token provided)
    
    Memory Management:
    - Garbage collection is forced after each processing step
    - Interim files are cleaned up after each file (unless --keep-interim is specified)
    - This ensures efficient memory usage when processing terabytes of data
    
    Batch Mode:
    - Use --batch-mode to process all h5ad files in a directory
    - Each file gets its own output directory and log file
    - Failures are logged but don't stop processing of other files
    - If token is provided, uploads happen after each successful file processing
    
    If --skip-train-test-split is used, the data will remain in a single parquet directory,
    allowing users on HuggingFace to decide on their own splitting strategy.
    """
    with start_action(action_type="cli_run_all", batch_mode=batch_mode, keep_interim=keep_interim):
        # Determine which files to process
        h5ad_files: list[Path] = []
        
        if h5ad_path is not None and h5ad_path.exists():
            if h5ad_path.is_dir():
                # Directory provided, process all h5ad files in it
                h5ad_files = sorted(list(h5ad_path.glob("*.h5ad")))
                typer.echo(f"Found {len(h5ad_files)} h5ad file(s) in {h5ad_path}")
            elif h5ad_path.suffix == ".h5ad":
                # Single file provided
                h5ad_files = [h5ad_path]
            else:
                typer.secho(f"Error: {h5ad_path} is not a valid h5ad file or directory", fg=typer.colors.RED)
                raise typer.Exit(1)
        else:
            # Auto-detect from input_dir
            h5ad_files = sorted(list(input_dir.glob("*.h5ad")))
            if h5ad_files:
                if batch_mode or len(h5ad_files) > 1:
                    typer.echo(f"Auto-detected {len(h5ad_files)} h5ad file(s) in {input_dir}")
                else:
                    # Single file, use only first one
                    h5ad_files = [h5ad_files[0]]
                    typer.echo(f"Auto-detected h5ad file: {h5ad_files[0]}")
            else:
                typer.secho(f"Error: Could not find h5ad file. Searched in {input_dir}", fg=typer.colors.RED)
                typer.echo("Please provide h5ad_path or download a dataset first using 'preprocess download'")
                raise typer.Exit(1)
        
        # Decide batch vs single mode
        process_multiple = batch_mode or len(h5ad_files) > 1
        
        # Step 1: Create HGNC mapper (optional, only if requested or if mappers_path not provided)
        mappers_path_final = mappers_path
        if create_hgnc:
            typer.echo("\n" + "="*80)
            typer.echo("STEP 1: Creating HGNC mapper")
            typer.echo("="*80)
            try:
                create_hgnc_mapper(interim_dir)
                typer.secho("✓ Step 1 complete\n", fg=typer.colors.GREEN)
                mappers_path_final = interim_dir / "hgnc_mappers.pkl"
            except Exception as e:
                typer.secho(f"⚠ Warning: Failed to create HGNC mapper: {e}", fg=typer.colors.YELLOW)
                typer.echo("Will proceed without HGNC mapper (will use gene symbols from h5ad if available)\n")
                mappers_path_final = None
        elif mappers_path is None:
            typer.echo("\n" + "="*80)
            typer.echo("STEP 1: Skipping HGNC mapper creation (use --create-hgnc to create)")
            typer.echo("="*80)
            typer.echo("HGNC will only be used if needed (AnnData has Ensembl IDs without gene symbols)\n")
            mappers_path_final = None
        
        # Process files
        results: list[tuple[str, bool, str]] = []
        
        for idx, h5ad_file in enumerate(h5ad_files, 1):
            dataset_name = sanitize_dataset_name(h5ad_file.stem)
            typer.echo("\n" + "="*80)
            if process_multiple:
                typer.echo(f"PROCESSING FILE {idx}/{len(h5ad_files)}: {dataset_name}")
            else:
                typer.echo(f"PROCESSING: {dataset_name}")
            typer.echo("="*80)
            
            # Setup per-file logging
            if log_dir:
                file_log = log_dir / dataset_name / "pipeline.log"
                file_log.parent.mkdir(parents=True, exist_ok=True)
                json_path = file_log.with_suffix('.json')
                to_nice_file(output_file=json_path, rendered_file=file_log)
            
            # Process the file
            success, message = _process_single_file(
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
            )
            
            results.append((dataset_name, success, message))
        
        # Summary
        typer.echo("\n" + "="*80)
        typer.secho("PIPELINE COMPLETE - SUMMARY", fg=typer.colors.GREEN, bold=True)
        typer.echo("="*80)
        
        successful = sum(1 for _, success, _ in results if success)
        failed = len(results) - successful
        
        typer.echo(f"Total files: {len(results)}")
        typer.echo(f"Successful: {successful}")
        if failed > 0:
            typer.echo(f"Failed: {failed}")
        
        typer.echo("\nDetails:")
        for dataset_name, success, message in results:
            status = "✓" if success else "✗"
            color = typer.colors.GREEN if success else typer.colors.RED
            typer.secho(f"  {status} {dataset_name}: {message}", fg=color)
        
        typer.echo(f"\nOutput directories:")
        typer.echo(f"  Input: {input_dir}")
        typer.echo(f"  Interim: {interim_dir}")
        typer.echo(f"  Output: {output_dir}")
        if log_dir:
            typer.echo(f"  Logs: {log_dir}")


def main() -> None:
    """Main entrypoint that shows help by default."""
    import sys
    # If no arguments provided, show help
    if len(sys.argv) == 1:
        sys.argv.append("--help")
    app()


if __name__ == "__main__":
    main()

