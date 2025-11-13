"""Integration tests to ensure streaming extraction matches historical output."""

from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from cell2sentence4longevity.explore import extract_fields_from_h5ad

DATASET_ID = "9deda9ad-6a71-401e-b909-5263919d85f9"
PROJECT_ROOT = Path(__file__).parent.parent
H5AD_PATH = PROJECT_ROOT / "data" / "input" / f"{DATASET_ID}.h5ad"
REFERENCE_SUMMARY = (
    Path(__file__).parent
    / "data"
    / "reference"
    / "explore"
    / f"{DATASET_ID}_meta_summary.csv"
)


@pytest.mark.integration
@pytest.mark.slow
def test_explore_extract_summary_matches_reference(tmp_path: Path) -> None:
    """Run explore.extract on a real dataset and compare summary with reference CSV."""
    if not H5AD_PATH.exists():
        pytest.skip(f"Required dataset missing: {H5AD_PATH}")
    if not REFERENCE_SUMMARY.exists():
        pytest.skip(f"Reference summary missing: {REFERENCE_SUMMARY}")

    output_dir = tmp_path / "meta"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{DATASET_ID}_meta.parquet"

    extract_fields_from_h5ad(
        h5ad_path=H5AD_PATH,
        fields=None,
        output_path=output_path,
        chunk_size=5000,
        compression="zstd",
        compression_level=3,
        use_pyarrow=True,
        extract_age=True,
        age_source_col="development_stage",
        generate_summary=True,
    )

    summary_path = output_path.with_name(f"{output_path.stem}_summary.csv")
    assert summary_path.exists(), "Summary CSV should be created"

    actual_df = pl.read_csv(summary_path)
    expected_df = pl.read_csv(REFERENCE_SUMMARY)

    assert actual_df.columns == expected_df.columns, "Summary columns should match reference"
    assert_frame_equal(actual_df, expected_df)

