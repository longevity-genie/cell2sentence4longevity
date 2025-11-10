"""Command-line interface for cell2sentence preprocessing pipeline."""

from pathlib import Path
from typing import Optional
import os

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
)

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
    mappers_path: Path = typer.Option(
        Path("./output/hgnc_mappers.pkl"),
        "--mappers",
        "-m",
        help="Path to HGNC mappers pickle file"
    ),
    output_dir: Path = typer.Option(
        Path("./output/temp_parquet"),
        "--output-dir",
        "-o",
        help="Directory to save parquet chunks"
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
def step3_add_age(
    parquet_dir: Path = typer.Option(
        Path("./output/temp_parquet"),
        "--parquet-dir",
        "-p",
        help="Directory containing parquet chunks"
    ),
    log_file: Optional[Path] = typer.Option(
        None,
        "--log-file",
        "-l",
        help="Path to eliot log file"
    ),
) -> None:
    """Step 3: Add age column and cleanup.
    
    Extracts age as integer from development_stage field and ensures proper column naming.
    """
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        json_path = log_file.with_suffix('.json')
        to_nice_file(output_file=json_path, rendered_file=log_file)
    
    with start_action(action_type="cli_step3_add_age"):
        typer.echo("Step 3: Adding age and cleaning up...")
        add_age_and_cleanup(parquet_dir)
        typer.secho("✓ Age added and cleanup completed", fg=typer.colors.GREEN)


@app.command()
def step4_train_test_split(
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
        create_train_test_split(parquet_dir, output_dir, test_size, random_state, chunk_size)
        typer.secho("✓ Train/test split created successfully", fg=typer.colors.GREEN)
        typer.echo(f"Output: {output_dir}")


@app.command()
def step5_upload(
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
    """Step 5: Upload to HuggingFace.
    
    Uploads the processed data to HuggingFace in a single batch commit.
    """
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        json_path = log_file.with_suffix('.json')
        to_nice_file(output_file=json_path, rendered_file=log_file)
    
    with start_action(action_type="cli_step5_upload"):
        typer.echo("Step 5: Uploading to HuggingFace...")
        upload_to_huggingface(data_splits_dir, token, repo_id, readme_path)
        typer.secho("✓ Upload completed successfully", fg=typer.colors.GREEN)
        typer.echo(f"Dataset: https://huggingface.co/datasets/{repo_id}")


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
    log_file: Optional[Path] = typer.Option(
        Path("./logs/pipeline.log"),
        "--log-file",
        "-l",
        help="Path to eliot log file"
    ),
) -> None:
    """Run all pipeline steps sequentially.
    
    This command runs all preprocessing steps in order:
    1. Create HGNC mapper
    2. Convert h5ad to parquet
    3. Add age and cleanup
    4. Create train/test split
    5. Upload to HuggingFace (if repo_id and token provided)
    """
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        json_path = log_file.with_suffix('.json')
        to_nice_file(output_file=json_path, rendered_file=log_file)
    
    with start_action(action_type="cli_run_all"):
        # Step 1
        typer.echo("\n" + "="*80)
        typer.echo("STEP 1: Creating HGNC mapper")
        typer.echo("="*80)
        create_hgnc_mapper(output_dir)
        typer.secho("✓ Step 1 complete\n", fg=typer.colors.GREEN)
        
        # Step 2
        typer.echo("="*80)
        typer.echo("STEP 2: Converting h5ad to parquet")
        typer.echo("="*80)
        mappers_path = output_dir / "hgnc_mappers.pkl"
        parquet_dir = output_dir / "temp_parquet"
        convert_h5ad_to_parquet(
            h5ad_path, mappers_path, parquet_dir, chunk_size, top_genes,
            compression=compression, compression_level=compression_level, use_pyarrow=use_pyarrow
        )
        typer.secho("✓ Step 2 complete\n", fg=typer.colors.GREEN)
        
        # Step 3
        typer.echo("="*80)
        typer.echo("STEP 3: Adding age and cleaning up")
        typer.echo("="*80)
        add_age_and_cleanup(parquet_dir)
        typer.secho("✓ Step 3 complete\n", fg=typer.colors.GREEN)
        
        # Step 4
        typer.echo("="*80)
        typer.echo("STEP 4: Creating train/test split")
        typer.echo("="*80)
        data_splits_dir = output_dir / "data_splits"
        create_train_test_split(parquet_dir, data_splits_dir, test_size, 42, chunk_size)
        typer.secho("✓ Step 4 complete\n", fg=typer.colors.GREEN)
        
        # Step 5 (optional)
        if repo_id and token:
            typer.echo("="*80)
            typer.echo("STEP 5: Uploading to HuggingFace")
            typer.echo("="*80)
            upload_to_huggingface(data_splits_dir, token, repo_id, None)
            typer.secho("✓ Step 5 complete\n", fg=typer.colors.GREEN)
            typer.echo(f"Dataset: https://huggingface.co/datasets/{repo_id}")
        else:
            typer.echo("\n⚠ Skipping Step 5 (upload) - token not provided")
        
        typer.echo("\n" + "="*80)
        typer.secho("✓ PIPELINE COMPLETE", fg=typer.colors.GREEN, bold=True)
        typer.echo("="*80)
        typer.echo(f"Output directory: {output_dir}")


if __name__ == "__main__":
    app()

