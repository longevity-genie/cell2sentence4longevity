"""Integration test ensuring preprocess streaming matches legacy output."""

from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from cell2sentence4longevity.preprocessing import convert_h5ad_to_train_test

DATASET_ID = "10cc50a0-af80-4fa1-b668-893dd5c0113a"
H5AD_PATH = Path("data/input") / f"{DATASET_ID}.h5ad"
REFERENCE_CHUNK = Path("tests/data/reference/preprocess/chunk_0000_reference.parquet")


@pytest.mark.integration
@pytest.mark.slow
def test_preprocess_chunk_matches_reference(tmp_path: Path) -> None:
    """Run preprocess conversion and ensure first test chunk matches reference output."""
    if not H5AD_PATH.exists():
        pytest.skip(f"Required dataset missing: {H5AD_PATH}")
    if not REFERENCE_CHUNK.exists():
        pytest.skip(f"Reference chunk missing: {REFERENCE_CHUNK}")

    output_dir = tmp_path / "output"
    convert_h5ad_to_train_test(
        h5ad_path=H5AD_PATH,
        output_dir=output_dir,
        dataset_name=DATASET_ID,
        chunk_size=5000,
        top_genes=2000,
        compression="zstd",
        compression_level=3,
        use_pyarrow=True,
        skip_train_test_split=False,
        stratify_by_age=True,
        join_collection=True,
        filter_by_age=True,
    )

    generated_chunk = output_dir / DATASET_ID / "test" / "chunk_0000.parquet"
    assert generated_chunk.exists(), "Generated test chunk should exist"

    expected_df = pl.read_parquet(REFERENCE_CHUNK)
    actual_df = pl.read_parquet(generated_chunk)

    # Align columns to guard against ordering differences.
    actual_df = actual_df.select(expected_df.columns)
    assert_frame_equal(actual_df, expected_df)

