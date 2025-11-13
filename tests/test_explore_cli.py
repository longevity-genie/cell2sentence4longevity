"""Tests for the explore CLI commands."""

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import polars as pl
import scipy.sparse as sp
from typer.testing import CliRunner

from cell2sentence4longevity import explore

runner = CliRunner()


def _write_dummy_h5ad(path: Path, n_cells: int = 5) -> None:
    """Create a small AnnData file for testing."""
    obs_index = [f"cell_{idx}" for idx in range(n_cells)]
    obs = {
        "development_stage": [f"{20 + idx} year-old" for idx in range(n_cells)],
        "organism": ["Homo sapiens"] * n_cells,
        "donor_id": [f"donor_{idx % 2}" for idx in range(n_cells)],
        "tissue": ["lung"] * n_cells,
        "cell_type": ["T cell"] * n_cells,
        "assay": ["10x"] * n_cells,
        "sex": ["female"] * n_cells,
        "disease": ["healthy"] * n_cells,
    }
    var = {"gene_ids": ["gene_a", "gene_b", "gene_c"]}
    rng = np.random.default_rng(seed=42)
    matrix = np.abs(rng.random((n_cells, len(var["gene_ids"]))))
    adata = ad.AnnData(
        X=sp.csr_matrix(matrix),
        obs=pd.DataFrame(obs, index=obs_index),
        var=pd.DataFrame(index=var["gene_ids"]),
    )
    adata.write_h5ad(path)


def test_explore_extract_creates_parquet(tmp_path: Path) -> None:
    """Ensure the extract command produces parquet output with age columns."""
    h5ad_path = tmp_path / "sample.h5ad"
    _write_dummy_h5ad(h5ad_path)

    output_path = tmp_path / "sample_meta.parquet"
    log_dir = tmp_path / "logs"

    result = runner.invoke(
        explore.app,
        [
            "extract",
            str(h5ad_path),
            "--output",
            str(output_path),
            "--log-dir",
            str(log_dir),
            "--summary",
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert output_path.exists()

    df = pl.read_parquet(output_path)
    assert df.height == 5
    assert "development_stage" in df.columns
    assert "age_years" in df.columns

    summary_path = output_path.with_name("sample_meta_summary.csv")
    assert summary_path.exists()


def test_explore_batch_processes_file(tmp_path: Path) -> None:
    """Verify batch command processes files and writes parquet output."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    log_dir = tmp_path / "logs"
    input_dir.mkdir()

    h5ad_path = input_dir / "dataset.h5ad"
    _write_dummy_h5ad(h5ad_path, n_cells=3)

    result = runner.invoke(
        explore.app,
        [
            "batch",
            str(input_dir),
            "--output-dir",
            str(output_dir),
            "--log-dir",
            str(log_dir),
            "--chunk-size",
            "2",
        ],
    )
    assert result.exit_code == 0, result.stdout

    parquet_path = output_dir / "dataset_meta.parquet"
    assert parquet_path.exists()

    df = pl.read_parquet(parquet_path)
    assert df.height == 3
    assert "cell_sentence" not in df.columns
    assert "development_stage" in df.columns


def test_explore_batch_respects_file_size_limit(tmp_path: Path) -> None:
    """Ensure batch command skips files exceeding the specified size limit."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    log_dir = tmp_path / "logs"
    input_dir.mkdir()

    h5ad_path = input_dir / "large_dataset.h5ad"
    _write_dummy_h5ad(h5ad_path, n_cells=2)

    result = runner.invoke(
        explore.app,
        [
            "batch",
            str(input_dir),
            "--output-dir",
            str(output_dir),
            "--log-dir",
            str(log_dir),
            "--max-file-size-mb",
            "0.0001",  # Force the file to be skipped
        ],
    )
    assert result.exit_code == 0, result.stdout

    # Command output should mention skipping due to size
    assert "Skipping" in result.stdout

    assert not list(output_dir.glob("*.parquet"))

