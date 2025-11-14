"""Tests for donor-level train/test split to prevent data leakage."""

import tempfile
from pathlib import Path

import numpy as np
import polars as pl
import pytest


def test_donor_split_no_leakage():
    """Test that donor-level split ensures no donor appears in both train and test."""
    
    # Create synthetic test data with multiple donors and cells
    np.random.seed(42)
    n_donors = 10
    cells_per_donor = 50
    
    # Generate donor data
    donors = [f"donor_{i}" for i in range(n_donors)]
    donor_ages = np.random.uniform(20, 80, n_donors)
    
    # Generate cell data
    data = []
    for donor_id, donor_age in zip(donors, donor_ages):
        for _ in range(cells_per_donor):
            # Add some variation in cell age (but centered on donor age)
            cell_age = donor_age + np.random.normal(0, 0.1)
            data.append({
                'donor_id': donor_id,
                'age': cell_age,
                'gene_sentence_2000': 'gene1 gene2 gene3',
                'cell_type': f'type_{np.random.randint(1, 5)}'
            })
    
    df = pl.DataFrame(data)
    
    # Write to temporary parquet files (simulating input chunks)
    with tempfile.TemporaryDirectory() as tmpdir:
        chunks_dir = Path(tmpdir) / 'chunks'
        chunks_dir.mkdir(parents=True, exist_ok=True)
        
        # Write in chunks
        chunk_size = 100
        n_chunks = (len(df) + chunk_size - 1) // chunk_size
        for i in range(n_chunks):
            start_idx = i * chunk_size
            end_idx = min(start_idx + chunk_size, len(df))
            chunk = df.slice(start_idx, end_idx - start_idx)
            chunk.write_parquet(chunks_dir / f"chunk_{i:04d}.parquet")
        
        # Run the donor-level split using our modified function
        from cell2sentence4longevity.preprocessing.train_test_split import create_train_test_split
        
        output_dir = Path(tmpdir) / 'output'
        create_train_test_split(
            parquet_dir=chunks_dir,
            output_dir=output_dir,
            dataset_name='test_dataset',
            test_size=0.2,
            random_state=42,
            chunk_size=100,
            compression='zstd',
            compression_level=3,
            use_pyarrow=True
        )
        
        # Load train and test sets
        train_dir = output_dir / 'test_dataset' / 'train'
        test_dir = output_dir / 'test_dataset' / 'test'
        
        train_df = pl.scan_parquet(train_dir / '*.parquet').collect()
        test_df = pl.scan_parquet(test_dir / '*.parquet').collect()
        
        # Verify no donor appears in both train and test
        train_donors = set(train_df['donor_id'].unique().to_list())
        test_donors = set(test_df['donor_id'].unique().to_list())
        
        overlap = train_donors & test_donors
        assert len(overlap) == 0, f"Found {len(overlap)} donors in both train and test: {overlap}"
        
        # Verify all donors are accounted for
        all_original_donors = set(df['donor_id'].unique().to_list())
        all_split_donors = train_donors | test_donors
        assert all_original_donors == all_split_donors, "Some donors were lost during split"
        
        # Verify test size is approximately correct at donor level
        donor_test_ratio = len(test_donors) / (len(train_donors) + len(test_donors))
        assert 0.1 <= donor_test_ratio <= 0.3, f"Test donor ratio {donor_test_ratio:.2f} is not close to target 0.2"
        
        # Verify all cells from same donor are in same split
        for donor in all_original_donors:
            donor_train_count = train_df.filter(pl.col('donor_id') == donor).height
            donor_test_count = test_df.filter(pl.col('donor_id') == donor).height
            assert (donor_train_count > 0 and donor_test_count == 0) or \
                   (donor_train_count == 0 and donor_test_count > 0), \
                   f"Donor {donor} has cells in both train ({donor_train_count}) and test ({donor_test_count})"
        
        print(f"✓ Donor-level split validation passed:")
        print(f"  - Train donors: {len(train_donors)}")
        print(f"  - Test donors: {len(test_donors)}")
        print(f"  - Train cells: {train_df.height}")
        print(f"  - Test cells: {test_df.height}")
        print(f"  - No donor leakage detected!")


def test_fallback_to_cell_level_split_without_donor_column():
    """Test that the split falls back to cell-level when no donor column exists."""
    
    # Create synthetic test data WITHOUT donor column
    np.random.seed(42)
    n_cells = 500
    
    data = []
    for i in range(n_cells):
        data.append({
            'age': np.random.uniform(20, 80),
            'gene_sentence_2000': 'gene1 gene2 gene3',
            'cell_type': f'type_{np.random.randint(1, 5)}'
        })
    
    df = pl.DataFrame(data)
    
    # Write to temporary parquet files
    with tempfile.TemporaryDirectory() as tmpdir:
        chunks_dir = Path(tmpdir) / 'chunks'
        chunks_dir.mkdir(parents=True, exist_ok=True)
        
        # Write in chunks
        chunk_size = 100
        n_chunks = (len(df) + chunk_size - 1) // chunk_size
        for i in range(n_chunks):
            start_idx = i * chunk_size
            end_idx = min(start_idx + chunk_size, len(df))
            chunk = df.slice(start_idx, end_idx - start_idx)
            chunk.write_parquet(chunks_dir / f"chunk_{i:04d}.parquet")
        
        # Run the split (should fall back to cell-level)
        from cell2sentence4longevity.preprocessing.train_test_split import create_train_test_split
        
        output_dir = Path(tmpdir) / 'output'
        create_train_test_split(
            parquet_dir=chunks_dir,
            output_dir=output_dir,
            dataset_name='test_dataset',
            test_size=0.2,
            random_state=42,
            chunk_size=100,
            compression='zstd',
            compression_level=3,
            use_pyarrow=True
        )
        
        # Load train and test sets
        train_dir = output_dir / 'test_dataset' / 'train'
        test_dir = output_dir / 'test_dataset' / 'test'
        
        train_df = pl.scan_parquet(train_dir / '*.parquet').collect()
        test_df = pl.scan_parquet(test_dir / '*.parquet').collect()
        
        # Verify split happened (basic sanity check)
        total_cells = train_df.height + test_df.height
        test_ratio = test_df.height / total_cells
        
        assert 0.15 <= test_ratio <= 0.25, f"Test ratio {test_ratio:.2f} is not close to target 0.2"
        
        print(f"✓ Cell-level split validation passed (fallback when no donor column):")
        print(f"  - Train cells: {train_df.height}")
        print(f"  - Test cells: {test_df.height}")
        print(f"  - Test ratio: {test_ratio:.2%}")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

