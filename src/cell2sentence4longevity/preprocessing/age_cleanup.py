"""Age extraction and column cleanup module."""

from pathlib import Path
import re

import polars as pl
from eliot import start_action
from tqdm import tqdm


def extract_age(development_stage: str | None) -> int | None:
    """Extract age in years from development_stage string.
    
    Args:
        development_stage: Development stage string (e.g., "22-year-old stage")
        
    Returns:
        Age in years or None if not found
    """
    if development_stage is None:
        return None
    
    # Match patterns like "22-year-old stage"
    match = re.search(r'(\d+)-year-old', str(development_stage))
    if match:
        return int(match.group(1))
    return None


def add_age_and_cleanup(parquet_dir: Path) -> None:
    """Add age column and cleanup column names.
    
    Args:
        parquet_dir: Directory containing parquet chunks
    """
    with start_action(action_type="add_age_and_cleanup", parquet_dir=str(parquet_dir)) as action:
        # Find all parquet files
        parquet_files = sorted(list(parquet_dir.glob("chunk_*.parquet")))
        action.log(message_type="found_parquet_files", count=len(parquet_files))
        
        # Check if age column already exists by checking schema (no data loading)
        schema = pl.scan_parquet(parquet_files[0]).collect_schema()
        needs_age = 'age' not in schema
        
        # Count total cells before processing
        lazy_dataset = pl.scan_parquet(parquet_dir / "chunk_*.parquet")
        total_cells_before = lazy_dataset.select(pl.count()).collect().item()
        action.log(message_type="total_cells_before_age_extraction", count=total_cells_before)
        
        if needs_age:
            action.log(message_type="adding_age_column", file_count=len(parquet_files))
            # Process and write back in chunks using streaming (sink_parquet)
            for filepath in tqdm(parquet_files, desc='Adding age'):
                # Load, transform, and collect
                df_with_age = pl.scan_parquet(filepath).with_columns(
                    pl.col('development_stage')
                    .map_elements(extract_age, return_dtype=pl.Int64)
                    .alias('age')
                ).collect()
                # Write back to same file
                df_with_age.write_parquet(filepath)
            action.log(message_type="age_column_added")
        
        # Check if cell2sentence exists and needs renaming (check schema, no data loading)
        if 'cell2sentence' in schema:
            action.log(message_type="renaming_column", from_name="cell2sentence", to_name="cell_sentence", file_count=len(parquet_files))
            for filepath in tqdm(parquet_files, desc='Renaming columns'):
                df_renamed = pl.scan_parquet(filepath).rename({'cell2sentence': 'cell_sentence'}).collect()
                df_renamed.write_parquet(filepath)
            action.log(message_type="column_renamed")
        else:
            action.log(message_type="column_naming_correct")
        
        # Verify final structure by checking schema
        final_schema = pl.scan_parquet(parquet_files[0]).collect_schema()
        action.log(
            message_type="verification_complete",
            total_columns=len(final_schema),
            has_age='age' in final_schema,
            has_cell_sentence='cell_sentence' in final_schema
        )
        
        if 'age' in final_schema:
            # Use lazy API for age distribution and null counting
            lazy_dataset = pl.scan_parquet(parquet_dir / "chunk_*.parquet")
            
            # Count cells with null age
            null_age_count = (
                lazy_dataset
                .filter(pl.col('age').is_null())
                .select(pl.count())
                .collect()
                .item()
            )
            
            # Count cells with valid age
            valid_age_count = (
                lazy_dataset
                .filter(pl.col('age').is_not_null())
                .select(pl.count())
                .collect()
                .item()
            )
            
            total_cells = null_age_count + valid_age_count
            
            action.log(
                message_type="age_extraction_summary",
                total_cells=total_cells,
                cells_with_valid_age=valid_age_count,
                cells_with_null_age=null_age_count,
                valid_age_percentage=round(valid_age_count / total_cells * 100, 2) if total_cells > 0 else 0,
                null_age_percentage=round(null_age_count / total_cells * 100, 2) if total_cells > 0 else 0
            )
            
            if null_age_count > 0:
                # Sample some null age cases to help debug
                sample_null_age = (
                    lazy_dataset
                    .filter(pl.col('age').is_null())
                    .select(['development_stage'])
                    .head(10)
                    .collect()
                )
                action.log(
                    message_type="sample_null_age_development_stages",
                    samples=sample_null_age.to_dicts()
                )
            
            # Get age distribution for valid ages
            age_counts = (
                lazy_dataset
                .filter(pl.col('age').is_not_null())
                .group_by('age')
                .agg(pl.count().alias('count'))
                .sort('age')
                .collect()
            )
            action.log(message_type="age_distribution", distribution=age_counts.to_dicts())

