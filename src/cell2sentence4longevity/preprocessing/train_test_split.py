"""Train/test split creation module."""

from pathlib import Path
import tempfile
import os

import polars as pl
from eliot import start_action
from tqdm import tqdm
from sklearn.model_selection import train_test_split as sklearn_split


def create_train_test_split(
    parquet_dir: Path,
    output_dir: Path,
    dataset_name: str | None = None,
    test_size: float = 0.05,
    random_state: int = 42,
    chunk_size: int | None = None,
    target_mb: float | None = None,
    compression: str = "zstd",
    compression_level: int = 3,
    use_pyarrow: bool = True
) -> None:
    """Create stratified train/test split.
    
    Args:
        parquet_dir: Directory containing parquet chunks (can be dataset_name/chunks/ or flat)
        output_dir: Directory to save train/test splits
        dataset_name: Name of the dataset for folder organization. If None, inferred from parquet_dir
        test_size: Proportion of data for test set
        random_state: Random seed for reproducibility
        chunk_size: Number of cells per output chunk (used if target_mb is None)
        target_mb: Target size per chunk in MB (used if chunk_size is None). Default: None (uses chunk_size=10000)
        compression: Compression algorithm for parquet files
        compression_level: Compression level
        use_pyarrow: Use pyarrow backend for parquet writes
    """
    # Determine dataset name and chunks directory
    # Check if parquet_dir ends with "chunks" (new structure) or contains chunk files (old structure)
    if (parquet_dir / "chunk_0000.parquet").exists() or list(parquet_dir.glob("chunk_*.parquet")):
        # Old flat structure or chunks directory
        chunks_dir = parquet_dir
        if dataset_name is None:
            # Try to infer from parent directory
            if parquet_dir.name == "chunks" and parquet_dir.parent.name:
                dataset_name = parquet_dir.parent.name
            else:
                dataset_name = "dataset"
    elif (parquet_dir.parent / "chunks" / "chunk_0000.parquet").exists():
        # New structure: parquet_dir is dataset_name, chunks are in dataset_name/chunks/
        chunks_dir = parquet_dir / "chunks"
        if dataset_name is None:
            dataset_name = parquet_dir.name
    else:
        # Assume it's the chunks directory
        chunks_dir = parquet_dir
        if dataset_name is None:
            dataset_name = "dataset"
    
    # Determine chunking strategy
    use_size_based = target_mb is not None
    if chunk_size is None and target_mb is None:
        chunk_size = 10000
        use_size_based = False
    elif chunk_size is not None and target_mb is not None:
        use_size_based = True
    
    with start_action(
        action_type="create_train_test_split",
        parquet_dir=str(parquet_dir),
        chunks_dir=str(chunks_dir),
        output_dir=str(output_dir),
        dataset_name=dataset_name,
        test_size=test_size,
        random_state=random_state,
        chunk_size=chunk_size,
        target_mb=target_mb,
        use_size_based=use_size_based
    ) as action:
        # Load all parquet files using lazy API - scan entire folder
        action.log(message_type="loading_parquet_chunks", chunks_dir=str(chunks_dir))
        
        # Use scan_parquet to read entire folder lazily
        lazy_dataset = pl.scan_parquet(chunks_dir / "chunk_*.parquet")
        
        # Count total cells before filtering
        total_cells_before = lazy_dataset.select(pl.len()).collect().item()
        action.log(message_type="total_cells_before_filtering", count=total_cells_before)
        
        # Check for null ages
        null_age_count = (
            lazy_dataset
            .filter(pl.col('age').is_null())
            .select(pl.len())
            .collect()
            .item()
        )
        
        if null_age_count > 0:
            action.log(
                message_type="cells_with_null_age_detected",
                null_age_count=null_age_count,
                null_age_percentage=round(null_age_count / total_cells_before * 100, 2)
            )
            
            # Sample some null age cases for debugging
            sample_null_age = (
                lazy_dataset
                .filter(pl.col('age').is_null())
                .select(['development_stage', 'age'])
                .head(10)
                .collect()
            )
            action.log(
                message_type="sample_cells_with_null_age",
                samples=sample_null_age.to_dicts()
            )
            
            # Filter out null ages
            action.log(message_type="filtering_null_ages")
            lazy_dataset = lazy_dataset.filter(pl.col('age').is_not_null())
        
        # Check age distribution using lazy API
        action.log(message_type="checking_age_distribution")
        age_counts = (
            lazy_dataset
            .group_by('age')
            .agg(pl.len().alias('count'))
            .sort('age')
            .collect()
        )
        action.log(message_type="age_distribution", distribution=age_counts.to_dicts())
        
        # Collect full dataset (only when needed for sklearn)
        action.log(message_type="loading_full_dataset")
        full_dataset = lazy_dataset.collect()
        total_cells_after_filtering = len(full_dataset)
        
        action.log(
            message_type="filtering_summary",
            cells_before_filtering=total_cells_before,
            cells_after_filtering=total_cells_after_filtering,
            cells_discarded=total_cells_before - total_cells_after_filtering,
            discard_percentage=round((total_cells_before - total_cells_after_filtering) / total_cells_before * 100, 2) if total_cells_before > 0 else 0
        )
        
        # Check for ages with too few samples for stratification
        min_samples_per_age = int(1 / test_size) if test_size > 0 else 2
        age_counts_dict = {row['age']: row['count'] for row in age_counts.to_dicts()}
        problematic_ages = {age: count for age, count in age_counts_dict.items() if count < min_samples_per_age}
        
        if problematic_ages:
            action.log(
                message_type="warning_ages_with_few_samples",
                min_samples_needed=min_samples_per_age,
                problematic_ages=problematic_ages,
                warning="Some ages have very few samples, stratification might fail"
            )
        
        # Shuffle dataset
        action.log(message_type="shuffling_dataset")
        full_dataset = full_dataset.sample(fraction=1.0, seed=random_state, shuffle=True)
        
        # Convert to pandas for sklearn compatibility
        action.log(message_type="creating_split")
        full_dataset_pd = full_dataset.to_pandas()
        
        # Stratified split by age
        train_df_pd, test_df_pd = sklearn_split(
            full_dataset_pd,
            test_size=test_size,
            stratify=full_dataset_pd['age'],
            random_state=random_state
        )
        
        # Convert back to Polars
        train_df = pl.from_pandas(train_df_pd)
        test_df = pl.from_pandas(test_df_pd)
        
        action.log(
            message_type="split_complete",
            train_cells=len(train_df),
            test_cells=len(test_df),
            train_pct=round(len(train_df) / total_cells_after_filtering * 100, 2),
            test_pct=round(len(test_df) / total_cells_after_filtering * 100, 2)
        )
        
        # Verify stratification
        train_age_counts = train_df.group_by('age').agg(pl.len().alias('count')).sort('age')
        test_age_counts = test_df.group_by('age').agg(pl.len().alias('count')).sort('age')
        action.log(
            message_type="train_age_distribution",
            distribution=train_age_counts.to_dicts()
        )
        action.log(
            message_type="test_age_distribution",
            distribution=test_age_counts.to_dicts()
        )
        
        # Save train and test sets
        action.log(message_type="saving_splits")
        
        # Create output directories: output_dir/dataset_name/train/chunks/ and output_dir/dataset_name/test/chunks/
        train_chunks_dir = output_dir / dataset_name / "train" / "chunks"
        test_chunks_dir = output_dir / dataset_name / "test" / "chunks"
        train_chunks_dir.mkdir(parents=True, exist_ok=True)
        test_chunks_dir.mkdir(parents=True, exist_ok=True)
        
        def save_chunks(df: pl.DataFrame, chunks_dir: Path, split_name: str) -> int:
            """Save DataFrame in chunks, returning number of chunks created."""
            chunk_idx = 0
            
            if use_size_based:
                target_bytes = int(target_mb * 1024 * 1024)
                batch_size = 5000
                
                row_idx = 0
                current_chunk_data = None
                
                while row_idx < len(df):
                    batch_end = min(row_idx + batch_size, len(df))
                    batch_df = df[row_idx:batch_end]
                    
                    if current_chunk_data is None:
                        current_chunk_data = batch_df
                    else:
                        current_chunk_data = pl.concat([current_chunk_data, batch_df])
                    
                    # Check size
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.parquet') as tmp_file:
                        temp_path = Path(tmp_file.name)
                        current_chunk_data.write_parquet(
                            temp_path,
                            compression=compression,
                            compression_level=compression_level,
                            use_pyarrow=use_pyarrow,
                        )
                        file_size = temp_path.stat().st_size
                        os.unlink(temp_path)
                    
                    if file_size >= target_bytes or batch_end == len(df):
                        output_file = chunks_dir / f"chunk_{chunk_idx:04d}.parquet"
                        current_chunk_data.write_parquet(
                            output_file,
                            compression=compression,
                            compression_level=compression_level,
                            use_pyarrow=use_pyarrow,
                        )
                        chunk_idx += 1
                        current_chunk_data = None
                    
                    row_idx = batch_end
            else:
                # Row-based chunking
                n_chunks = (len(df) + chunk_size - 1) // chunk_size
                for i in range(n_chunks):
                    start_idx = i * chunk_size
                    end_idx = min(start_idx + chunk_size, len(df))
                    chunk = df[start_idx:end_idx]
                    chunk.write_parquet(
                        chunks_dir / f"chunk_{i:04d}.parquet",
                        compression=compression,
                        compression_level=compression_level,
                        use_pyarrow=use_pyarrow,
                    )
                chunk_idx = n_chunks
            
            return chunk_idx
        
        # Save train set in chunks
        action.log(message_type="saving_train_chunks")
        n_train_chunks = save_chunks(train_df, train_chunks_dir, "train")
        
        # Save test set in chunks
        action.log(message_type="saving_test_chunks")
        n_test_chunks = save_chunks(test_df, test_chunks_dir, "test")
        
        # Calculate sizes
        train_size = sum(f.stat().st_size for f in train_chunks_dir.glob("*.parquet"))
        test_size_bytes = sum(f.stat().st_size for f in test_chunks_dir.glob("*.parquet"))
        
        action.log(
            message_type="split_saved",
            train_chunks=n_train_chunks,
            test_chunks=n_test_chunks,
            train_size_gb=round(train_size / (1024**3), 3),
            test_size_gb=round(test_size_bytes / (1024**3), 3),
            train_chunks_dir=str(train_chunks_dir),
            test_chunks_dir=str(test_chunks_dir)
        )

