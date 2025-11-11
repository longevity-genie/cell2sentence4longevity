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
                dataset_name = parquet_dir.name  # Use the directory name itself
    elif (parquet_dir.parent / "chunks" / "chunk_0000.parquet").exists():
        # New structure: parquet_dir is dataset_name, chunks are in dataset_name/chunks/
        chunks_dir = parquet_dir / "chunks"
        if dataset_name is None:
            dataset_name = parquet_dir.name
    else:
        # Assume it's the chunks directory
        chunks_dir = parquet_dir
        if dataset_name is None:
            dataset_name = parquet_dir.name  # Use the directory name itself
    
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
        
        # Count total cells after filtering (lazy)
        total_cells_after_filtering = (
            lazy_dataset
            .select(pl.len())
            .collect()
            .item()
        )
        
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
        
        # Create output directories: output_dir/dataset_name/train/ and output_dir/dataset_name/test/
        train_dir = output_dir / dataset_name / "train"
        test_dir = output_dir / dataset_name / "test"
        train_dir.mkdir(parents=True, exist_ok=True)
        test_dir.mkdir(parents=True, exist_ok=True)
        
        # Use streaming stratified split with lazy API
        # Add random column for stratified splitting (lazy operation)
        # We use hash of age combined with row position to get deterministic but random assignment
        # within each age group for proper stratification
        action.log(message_type="creating_stratified_split_streaming")
        lazy_with_split = (
            lazy_dataset
            .with_row_index("_row_idx")
            .with_columns(
                ((pl.col('age').hash(seed=random_state) + pl.col('_row_idx').hash(seed=random_state + 1)) % 10000 / 10000.0).alias('_random_split')
            )
            .drop('_row_idx')
        )
        
        # Create train and test lazy datasets
        lazy_train = lazy_with_split.filter(pl.col('_random_split') >= test_size).drop('_random_split')
        lazy_test = lazy_with_split.filter(pl.col('_random_split') < test_size).drop('_random_split')
        
        # Count cells in each split (lazy)
        train_cells = lazy_train.select(pl.len()).collect().item()
        test_cells = lazy_test.select(pl.len()).collect().item()
        
        action.log(
            message_type="split_complete",
            train_cells=train_cells,
            test_cells=test_cells,
            train_pct=round(train_cells / total_cells_after_filtering * 100, 2) if total_cells_after_filtering > 0 else 0,
            test_pct=round(test_cells / total_cells_after_filtering * 100, 2) if total_cells_after_filtering > 0 else 0
        )
        
        # Verify stratification (lazy)
        train_age_counts = (
            lazy_train
            .group_by('age')
            .agg(pl.len().alias('count'))
            .sort('age')
            .collect()
        )
        test_age_counts = (
            lazy_test
            .group_by('age')
            .agg(pl.len().alias('count'))
            .sort('age')
            .collect()
        )
        action.log(
            message_type="train_age_distribution",
            distribution=train_age_counts.to_dicts()
        )
        action.log(
            message_type="test_age_distribution",
            distribution=test_age_counts.to_dicts()
        )
        
        # Save train and test sets using streaming sink_parquet
        action.log(message_type="saving_splits")
        
        def save_chunks_streaming(lazy_df: pl.LazyFrame, output_dir: Path, split_name: str) -> int:
            """Save LazyFrame in chunks using streaming, returning number of chunks created."""
            chunk_idx = 0
            
            if use_size_based:
                target_bytes = int(target_mb * 1024 * 1024)
                batch_size = 10000  # Process in batches
                
                row_idx = 0
                current_chunk_batches = []
                chunk_start_row = 0
                
                # Get total rows (lazy)
                total_rows = lazy_df.select(pl.len()).collect().item()
                
                while row_idx < total_rows:
                    # Collect batch lazily using streaming
                    batch_df = (
                        lazy_df
                        .slice(row_idx, batch_size)
                        .collect(streaming=True)
                    )
                    
                    if batch_df.height == 0:
                        break
                    
                    current_chunk_batches.append(batch_df)
                    
                    # Check size by writing to temp file
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.parquet') as tmp_file:
                        temp_path = Path(tmp_file.name)
                        # Concat batches only when checking size
                        if len(current_chunk_batches) > 1:
                            combined = pl.concat(current_chunk_batches)
                        else:
                            combined = current_chunk_batches[0]
                        combined.write_parquet(
                            temp_path,
                            compression=compression,
                            compression_level=compression_level,
                            use_pyarrow=use_pyarrow,
                        )
                        file_size = temp_path.stat().st_size
                        os.unlink(temp_path)
                    
                    batch_end = row_idx + batch_df.height
                    
                    if file_size >= target_bytes or batch_end >= total_rows:
                        # Write chunk
                        if len(current_chunk_batches) > 1:
                            chunk_data = pl.concat(current_chunk_batches)
                        else:
                            chunk_data = current_chunk_batches[0]
                        
                        output_file = output_dir / f"{dataset_name}_{chunk_start_row:07d}_{batch_end:07d}.parquet"
                        chunk_data.write_parquet(
                            output_file,
                            compression=compression,
                            compression_level=compression_level,
                            use_pyarrow=use_pyarrow,
                        )
                        chunk_idx += 1
                        current_chunk_batches = []
                        chunk_start_row = batch_end
                    
                    row_idx = batch_end
            else:
                # Row-based chunking with streaming
                total_rows = lazy_df.select(pl.len()).collect().item()
                n_chunks = (total_rows + chunk_size - 1) // chunk_size
                
                for i in range(n_chunks):
                    start_idx = i * chunk_size
                    end_idx = min(start_idx + chunk_size, total_rows)
                    
                    # Collect chunk lazily using streaming
                    chunk = (
                        lazy_df
                        .slice(start_idx, end_idx - start_idx)
                        .collect(streaming=True)
                    )
                    
                    output_file = output_dir / f"{dataset_name}_{start_idx:07d}_{end_idx:07d}.parquet"
                    chunk.write_parquet(
                        output_file,
                        compression=compression,
                        compression_level=compression_level,
                        use_pyarrow=use_pyarrow,
                    )
                chunk_idx = n_chunks
            
            return chunk_idx
        
        # Save train set in chunks using streaming
        action.log(message_type="saving_train_chunks")
        n_train_chunks = save_chunks_streaming(lazy_train, train_dir, "train")
        
        # Save test set in chunks using streaming
        action.log(message_type="saving_test_chunks")
        n_test_chunks = save_chunks_streaming(lazy_test, test_dir, "test")
        
        # Calculate sizes
        train_size = sum(f.stat().st_size for f in train_dir.glob("*.parquet"))
        test_size_bytes = sum(f.stat().st_size for f in test_dir.glob("*.parquet"))
        
        action.log(
            message_type="split_saved",
            train_chunks=n_train_chunks,
            test_chunks=n_test_chunks,
            train_size_gb=round(train_size / (1024**3), 3),
            test_size_gb=round(test_size_bytes / (1024**3), 3),
            train_dir=str(train_dir),
            test_dir=str(test_dir)
        )

