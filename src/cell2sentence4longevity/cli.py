"""Command-line interface for cell2sentence preprocessing pipeline."""

from pathlib import Path
from typing import Optional
import os
import time

import typer
from eliot import start_action
from pycomfort.logging import to_nice_file
from dotenv import load_dotenv

from cell2sentence4longevity.preprocessing import (
    create_hgnc_mapper,
    convert_h5ad_to_parquet,
    create_train_test_split,
    upload_to_huggingface,
)
from cell2sentence4longevity.cleanup import cleanup_old_tests

# Load environment variables from .env file
load_dotenv()

app = typer.Typer(
    name="cell2sentence",
    help="Cell to sentence preprocessing pipeline for longevity research",
    add_completion=False,
)


@app.command()
def step1_hgnc_mapper(
    output_dir: Path = typer.Option(
        Path("./output"),
        "--output-dir",
        "-o",
        help="Directory to save HGNC mappers"
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
        create_hgnc_mapper(output_dir)
        typer.secho("✓ HGNC mapper created successfully", fg=typer.colors.GREEN)
        typer.echo(f"Output: {output_dir / 'hgnc_mappers.pkl'}")


@app.command()
def step2_convert_h5ad(
    h5ad_path: Path = typer.Argument(
        ...,
        help="Path to AIDA h5ad file"
    ),
    mappers_path: Path | None = typer.Option(
        None,
        "--mappers",
        "-m",
        help="Path to HGNC mappers pickle file (optional, only used if needed or explicitly provided)"
    ),
    output_dir: Path = typer.Option(
        Path("./output/temp_parquet"),
        "--output-dir",
        "-o",
        help="Directory to save parquet chunks"
    ),
    chunk_size: int = typer.Option(
        2500,
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
    log_file: Optional[Path] = typer.Option(
        None,
        "--log-file",
        "-l",
        help="Path to eliot log file"
    ),
) -> None:
    """Step 2: Convert h5ad to parquet with cell sentences.
    
    Transforms the AIDA h5ad file into parquet chunks, creating "cell sentences" -
    space-separated gene symbols ordered by expression level.
    """
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        json_path = log_file.with_suffix('.json')
        to_nice_file(output_file=json_path, rendered_file=log_file)
    
    with start_action(action_type="cli_step2_convert_h5ad"):
        typer.echo("Step 2: Converting h5ad to parquet...")
        convert_h5ad_to_parquet(
            h5ad_path, mappers_path, output_dir, chunk_size, top_genes,
            compression=compression, compression_level=compression_level, use_pyarrow=use_pyarrow
        )
        typer.secho("✓ Conversion completed successfully", fg=typer.colors.GREEN)
        typer.echo(f"Output: {output_dir}")


@app.command()
def step3_train_test_split(
    parquet_dir: Path = typer.Option(
        Path("./output/temp_parquet"),
        "--parquet-dir",
        "-p",
        help="Directory containing parquet chunks"
    ),
    output_dir: Path = typer.Option(
        Path("./output/data_splits"),
        "--output-dir",
        "-o",
        help="Directory to save train/test splits"
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
        2500,
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
    """Step 3: Create stratified train/test split.
    
    Creates a train/test split stratified by age to maintain age distribution in both sets.
    """
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        json_path = log_file.with_suffix('.json')
        to_nice_file(output_file=json_path, rendered_file=log_file)
    
    with start_action(action_type="cli_step3_train_test_split"):
        typer.echo("Step 3: Creating train/test split...")
        create_train_test_split(parquet_dir, output_dir, test_size, random_state, chunk_size)
        typer.secho("✓ Train/test split created successfully", fg=typer.colors.GREEN)
        typer.echo(f"Output: {output_dir}")


@app.command()
def step4_upload(
    data_splits_dir: Path = typer.Option(
        Path("./output/data_splits"),
        "--data-splits-dir",
        "-d",
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
    """Step 4: Upload to HuggingFace.
    
    Uploads the processed data to HuggingFace in a single batch commit.
    """
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        json_path = log_file.with_suffix('.json')
        to_nice_file(output_file=json_path, rendered_file=log_file)
    
    with start_action(action_type="cli_step4_upload"):
        typer.echo("Step 4: Uploading to HuggingFace...")
        upload_to_huggingface(data_splits_dir, token, repo_id, readme_path)
        typer.secho("✓ Upload completed successfully", fg=typer.colors.GREEN)
        typer.echo(f"Dataset: https://huggingface.co/datasets/{repo_id}")


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
    with start_action(action_type="cli_cleanup"):
        typer.echo("Cleaning up old test directories...")
        cleanup_old_tests(days)
        typer.secho("✓ Cleanup completed", fg=typer.colors.GREEN)


@app.command()
def run_all(
    h5ad_path: Path = typer.Argument(
        ...,
        help="Path to AIDA h5ad file"
    ),
    output_dir: Path = typer.Option(
        Path("./output"),
        "--output-dir",
        "-o",
        help="Base output directory"
    ),
    repo_id: Optional[str] = typer.Option(
        "longevity-genie/cell2sentence4longevity-data",
        "--repo-id",
        "-r",
        help="HuggingFace repository ID. Defaults to 'longevity-genie/cell2sentence4longevity-data'"
    ),
    token: Optional[str] = typer.Option(
        None,
        "--token",
        "-t",
        help="HuggingFace API token (skip step 5 if not provided)",
        envvar="HF_TOKEN"
    ),
    chunk_size: int = typer.Option(
        2500,
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
    log_file: Optional[Path] = typer.Option(
        Path("./logs/pipeline.log"),
        "--log-file",
        "-l",
        help="Path to eliot log file"
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
    2. Convert h5ad to parquet (includes age extraction during conversion)
    3. Create train/test split (optional, can be skipped with --skip-train-test-split)
    4. Upload to HuggingFace (if repo_id and token provided)
    
    If --skip-train-test-split is used, the data will remain in a single parquet directory,
    allowing users on HuggingFace to decide on their own splitting strategy.
    """
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        json_path = log_file.with_suffix('.json')
        to_nice_file(output_file=json_path, rendered_file=log_file)
    
    with start_action(action_type="cli_run_all") as action:
        # Start timing (excluding download - assuming h5ad_path already exists)
        pipeline_start_time = time.time()
        
        # Step 1: Create HGNC mapper (optional)
        mappers_path_final = mappers_path
        if create_hgnc:
            typer.echo("\n" + "="*80)
            typer.echo("STEP 1: Creating HGNC mapper")
            typer.echo("="*80)
            try:
                create_hgnc_mapper(output_dir)
                typer.secho("✓ Step 1 complete\n", fg=typer.colors.GREEN)
                mappers_path_final = output_dir / "hgnc_mappers.pkl"
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
        
        # Step 2
        typer.echo("="*80)
        typer.echo("STEP 2: Converting h5ad to parquet")
        typer.echo("="*80)
        parquet_dir = output_dir / "temp_parquet"
        convert_h5ad_to_parquet(
            h5ad_path, mappers_path_final, parquet_dir, chunk_size, top_genes,
            compression=compression, compression_level=compression_level, use_pyarrow=use_pyarrow
        )
        typer.secho("✓ Step 2 complete\n", fg=typer.colors.GREEN)
        
        # Step 3 (optional, based on skip_train_test_split flag)
        if skip_train_test_split:
            typer.echo("\n⚠ Skipping Step 3 (train/test split) - data will remain in single parquet directory")
            # Data for upload is in parquet_dir
            upload_dir = parquet_dir
        else:
            typer.echo("="*80)
            typer.echo("STEP 3: Creating train/test split")
            typer.echo("="*80)
            data_splits_dir = output_dir / "data_splits"
            create_train_test_split(parquet_dir, data_splits_dir, test_size, 42, chunk_size)
            typer.secho("✓ Step 3 complete\n", fg=typer.colors.GREEN)
            # Data for upload is in data_splits_dir
            upload_dir = data_splits_dir
        
        # Step 4 (optional)
        if repo_id and token:
            typer.echo("="*80)
            typer.echo("STEP 4: Uploading to HuggingFace")
            typer.echo("="*80)
            upload_to_huggingface(upload_dir, token, repo_id, None)
            typer.secho("✓ Step 4 complete\n", fg=typer.colors.GREEN)
            typer.echo(f"Dataset: https://huggingface.co/datasets/{repo_id}")
        else:
            typer.echo("\n⚠ Skipping Step 4 (upload) - token not provided")
        
        # Calculate and log total execution time
        pipeline_end_time = time.time()
        total_execution_time = pipeline_end_time - pipeline_start_time
        
        # Format time as hh:mm:ss
        hours = int(total_execution_time // 3600)
        minutes = int((total_execution_time % 3600) // 60)
        seconds = int(total_execution_time % 60)
        time_formatted = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        
        # Log execution time
        action.log(
            message_type="pipeline_execution_time",
            total_seconds=total_execution_time,
            formatted_time=time_formatted,
            note="Excluding download time"
        )
        
        typer.echo("\n" + "="*80)
        typer.secho("✓ PIPELINE COMPLETE", fg=typer.colors.GREEN, bold=True)
        typer.echo("="*80)
        typer.echo(f"Output directory: {output_dir}")
        typer.echo(f"Total execution time (excluding download): {time_formatted} ({total_execution_time:.2f} seconds)")
        typer.echo("="*80)


if __name__ == "__main__":
    app()

