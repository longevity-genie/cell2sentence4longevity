"""Train/test split creation module."""

from pathlib import Path

import polars as pl
from eliot import start_action
from tqdm import tqdm
from sklearn.model_selection import train_test_split as sklearn_split


def create_train_test_split(
    parquet_dir: Path,
    output_dir: Path,
    test_size: float = 0.05,
    random_state: int = 42,
    chunk_size: int = 10000
) -> None:
    """Create stratified train/test split.
    
    Args:
        parquet_dir: Directory containing parquet chunks
        output_dir: Directory to save train/test splits
        test_size: Proportion of data for test set
        random_state: Random seed for reproducibility
        chunk_size: Number of cells per output chunk
    """
    with start_action(
        action_type="create_train_test_split",
        parquet_dir=str(parquet_dir),
        output_dir=str(output_dir),
        test_size=test_size,
        random_state=random_state
    ) as action:
        # Load all parquet files using lazy API - scan entire folder
        action.log(message_type="loading_parquet_chunks")
        
        # Use scan_parquet to read entire folder lazily
        lazy_dataset = pl.scan_parquet(parquet_dir / "chunk_*.parquet")
        
        # Count total cells before filtering
        total_cells_before = lazy_dataset.select(pl.count()).collect().item()
        action.log(message_type="total_cells_before_filtering", count=total_cells_before)
        
        # Check for null ages
        null_age_count = (
            lazy_dataset
            .filter(pl.col('age').is_null())
            .select(pl.count())
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
            .agg(pl.count().alias('count'))
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
        train_age_counts = train_df.group_by('age').agg(pl.count().alias('count')).sort('age')
        test_age_counts = test_df.group_by('age').agg(pl.count().alias('count')).sort('age')
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
        
        # Create output directories
        train_dir = output_dir / "train"
        test_dir = output_dir / "test"
        train_dir.mkdir(parents=True, exist_ok=True)
        test_dir.mkdir(parents=True, exist_ok=True)
        
        # Save train set in chunks
        n_train_chunks = (len(train_df) + chunk_size - 1) // chunk_size
        action.log(message_type="saving_train_chunks", n_chunks=n_train_chunks)
        
        for i in tqdm(range(n_train_chunks), desc='Train chunks'):
            start_idx = i * chunk_size
            end_idx = min(start_idx + chunk_size, len(train_df))
            chunk = train_df[start_idx:end_idx]
            chunk.write_parquet(train_dir / f"chunk_{i:04d}.parquet")
        
        # Save test set in chunks
        n_test_chunks = (len(test_df) + chunk_size - 1) // chunk_size
        action.log(message_type="saving_test_chunks", n_chunks=n_test_chunks)
        
        for i in tqdm(range(n_test_chunks), desc='Test chunks'):
            start_idx = i * chunk_size
            end_idx = min(start_idx + chunk_size, len(test_df))
            chunk = test_df[start_idx:end_idx]
            chunk.write_parquet(test_dir / f"chunk_{i:04d}.parquet")
        
        # Calculate sizes
        train_size = sum(f.stat().st_size for f in train_dir.glob("*.parquet"))
        test_size_bytes = sum(f.stat().st_size for f in test_dir.glob("*.parquet"))
        
        action.log(
            message_type="split_saved",
            train_chunks=n_train_chunks,
            test_chunks=n_test_chunks,
            train_size_gb=round(train_size / (1024**3), 3),
            test_size_gb=round(test_size_bytes / (1024**3), 3)
        )

