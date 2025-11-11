"""H5AD to Parquet converter with cell sentence generation."""

from pathlib import Path
from typing import Dict, List, Optional
import pickle
import time
import tempfile
import os
import re

import anndata as ad
import numpy as np
from eliot import start_action
from tqdm import tqdm
import polars as pl


def extract_age_column(df: pl.DataFrame, source_col: str = 'development_stage') -> pl.DataFrame:
    """Extract age in years from development_stage column using Polars regex.
    
    This is much faster than using map_elements as it's fully vectorized.
    
    Args:
        df: Polars DataFrame with development_stage column
        source_col: Name of the column containing development stage info
        
    Returns:
        DataFrame with added 'age' column (Float64)
    """
    if source_col not in df.columns:
        # No source column, create null age column
        return df.with_columns(pl.lit(None).cast(pl.Float64).alias('age'))
    
    # Use Polars regex to extract age - much faster than map_elements
    # Pattern matches "22-year-old" or "22.5-year-old"
    return df.with_columns(
        pl.col(source_col)
        .str.extract(r'(\d+(?:\.\d+)?)-year-old', 1)  # Extract first capture group
        .cast(pl.Float64, strict=False)  # Cast to float, null if not a number
        .alias('age')
    )


def has_gene_symbols(adata: ad.AnnData) -> bool:
    """Check if AnnData var_names appear to be gene symbols rather than Ensembl IDs.
    
    Args:
        adata: AnnData object
        
    Returns:
        True if var_names appear to be gene symbols, False if they look like Ensembl IDs
    """
    if adata.n_vars == 0:
        return False
    
    # Check a sample of var_names to determine if they're Ensembl IDs
    # Ensembl IDs typically start with "ENS" (e.g., ENSG00000139618)
    sample_size = min(100, adata.n_vars)
    sample_names = [adata.var_names[i] for i in range(sample_size)]
    
    # Count how many look like Ensembl IDs
    ensembl_like_count = sum(1 for name in sample_names if name.startswith("ENS"))
    
    # If more than 50% look like Ensembl IDs, assume they are Ensembl IDs
    return ensembl_like_count < (sample_size * 0.5)


def feature_name_has_gene_symbols(adata: ad.AnnData) -> bool:
    """Check if AnnData feature_name column contains gene symbols rather than Ensembl IDs.
    
    Args:
        adata: AnnData object
        
    Returns:
        True if feature_name contains gene symbols, False otherwise
    """
    if 'feature_name' not in adata.var.columns:
        return False
    
    if adata.n_vars == 0:
        return False
    
    # Check a sample of feature_names to determine if they're Ensembl IDs
    sample_size = min(100, adata.n_vars)
    sample_features = [adata.var['feature_name'].iloc[i] for i in range(sample_size)]
    
    # Count how many look like Ensembl IDs
    ensembl_like_count = sum(1 for name in sample_features if str(name).startswith("ENS"))
    
    # If more than 50% look like Ensembl IDs, assume feature_name doesn't have gene symbols
    return ensembl_like_count < (sample_size * 0.5)


def load_h5ad_data(h5ad_path: Path) -> ad.AnnData:
    """Load AIDA h5ad file in backed mode.
    
    Args:
        h5ad_path: Path to the h5ad file
        
    Returns:
        AnnData object in backed mode
    """
    with start_action(action_type="load_h5ad_data", path=str(h5ad_path)) as action:
        adata = ad.read_h5ad(h5ad_path, backed='r')
        action.log(message_type="h5ad_loaded", n_cells=adata.n_obs, n_genes=adata.n_vars)
        
        # Check for critical columns without loading full metadata into memory
        # Access columns directly from the backed object
        has_development_stage = 'development_stage' in adata.obs.columns
        has_feature_name = 'feature_name' in adata.var.columns
        
        # Log metadata structure without loading everything into memory
        n_obs_columns = len(adata.obs.columns)
        n_var_columns = len(adata.var.columns)
        
        action.log(
            message_type="metadata_structure",
            n_obs_columns=n_obs_columns,
            n_var_columns=n_var_columns
        )
        
        action.log(
            message_type="critical_columns_check",
            has_development_stage=has_development_stage,
            has_feature_name=has_feature_name
        )
        
        if not has_development_stage:
            action.log(
                message_type="warning_missing_column",
                column="development_stage",
                warning="Age extraction will not be possible without development_stage column"
            )
        
        if not has_feature_name:
            action.log(
                message_type="warning_missing_column",
                column="feature_name",
                warning="Gene mapping fallback will not be available without feature_name column"
            )
        
        return adata


def map_genes_to_symbols_polars(
    adata: ad.AnnData,
    hgnc_df: pl.DataFrame | None = None,
    use_hgnc: bool = True
) -> List[str]:
    """Map Ensembl IDs to gene symbols using Polars for fast joins.
    
    Args:
        adata: AnnData object
        hgnc_df: HGNC DataFrame with mapping data (only used if use_hgnc=True)
        use_hgnc: Whether to use HGNC mapping (True) or use var_names directly (False)
        
    Returns:
        List of gene symbols (Ensembl IDs are filtered out during sentence creation)
    """
    with start_action(action_type="map_genes_to_symbols_polars", use_hgnc=use_hgnc) as action:
        n_genes = adata.n_vars
        action.log(message_type="starting_gene_mapping", total_genes=n_genes, use_hgnc=use_hgnc)
        
        if not use_hgnc:
            # Check if we should use feature_name instead of var_names
            has_feature_name = 'feature_name' in adata.var.columns
            use_feature_name = False
            
            if has_feature_name:
                # Sample a few feature_names to check if they're gene symbols
                sample_size = min(100, n_genes)
                sample_features = [adata.var['feature_name'].iloc[i] for i in range(sample_size)]
                feature_ensembl_count = sum(1 for name in sample_features if str(name).startswith("ENS"))
                
                # If feature_name has fewer Ensembl IDs than var_names, prefer feature_name
                if feature_ensembl_count < (sample_size * 0.5):
                    use_feature_name = True
                    action.log(message_type="using_feature_name_column", reason="contains_gene_symbols")
            
            if use_feature_name:
                gene_symbols = [str(adata.var['feature_name'].iloc[i]) for i in range(n_genes)]
            else:
                gene_symbols = list(adata.var_names)
            
            action.log(
                message_type="gene_mapping_summary",
                total_genes=len(gene_symbols),
                used_feature_name=use_feature_name
            )
            return gene_symbols
        
        # Use HGNC mapping with Polars join (much faster than dict lookup!)
        action.log(message_type="preparing_polars_join")
        
        # Normalize HGNC column names if needed
        col_map = {
            'Ensembl gene ID': 'ensembl_gene_id',
            'Approved symbol': 'symbol'
        }
        for old_name, new_name in col_map.items():
            if old_name in hgnc_df.columns:
                hgnc_df = hgnc_df.rename({old_name: new_name})
        
        # Create DataFrame from adata with var_names and feature_name
        has_feature_name = 'feature_name' in adata.var.columns
        
        if has_feature_name:
            genes_df = pl.DataFrame({
                'ensembl_gene_id': list(adata.var_names),
                'feature_name': [str(adata.var['feature_name'].iloc[i]) for i in range(n_genes)],
                'original_index': list(range(n_genes))
            })
        else:
            genes_df = pl.DataFrame({
                'ensembl_gene_id': list(adata.var_names),
                'original_index': list(range(n_genes))
            })
        
        # Join with HGNC to get official symbols
        action.log(message_type="performing_hgnc_join")
        hgnc_mapping = hgnc_df.select(['ensembl_gene_id', 'symbol']).filter(
            pl.col('ensembl_gene_id').is_not_null() & pl.col('symbol').is_not_null()
        )
        
        # Deduplicate hgnc_mapping to avoid creating duplicate rows in join
        # If multiple symbols exist for same ensembl_gene_id, keep the first one
        hgnc_mapping = hgnc_mapping.unique(subset=['ensembl_gene_id'], keep='first')
        
        # Left join to keep all genes
        mapped_df = genes_df.join(hgnc_mapping, on='ensembl_gene_id', how='left')
        
        # Determine final gene symbol with fallback logic
        if has_feature_name:
            # Priority: 1) HGNC symbol, 2) feature_name (if not Ensembl ID), 3) ensembl_gene_id
            mapped_df = mapped_df.with_columns([
                pl.when(pl.col('symbol').is_not_null())
                .then(pl.col('symbol'))
                .when(~pl.col('feature_name').str.starts_with("ENS"))
                .then(pl.col('feature_name'))
                .otherwise(pl.col('ensembl_gene_id'))
                .alias('final_symbol')
            ])
        else:
            # Priority: 1) HGNC symbol, 2) ensembl_gene_id
            mapped_df = mapped_df.with_columns([
                pl.when(pl.col('symbol').is_not_null())
                .then(pl.col('symbol'))
                .otherwise(pl.col('ensembl_gene_id'))
                .alias('final_symbol')
            ])
        
        # Sort by original index to maintain order
        mapped_df = mapped_df.sort('original_index')
        
        # Ensure we have exactly one row per original_index (deduplicate if join created duplicates)
        if mapped_df.height != n_genes:
            action.log(
                message_type="warning_duplicate_rows_after_join",
                mapped_df_height=mapped_df.height,
                expected_height=n_genes,
                note="Deduplicating by original_index, keeping first occurrence"
            )
            mapped_df = mapped_df.unique(subset=['original_index'], keep='first')
        
        # Extract gene symbols list
        gene_symbols = mapped_df['final_symbol'].to_list()
        
        # Final validation: ensure we have the correct number of genes
        if len(gene_symbols) != n_genes:
            raise ValueError(
                f"Gene mapping length mismatch: expected {n_genes} genes, "
                f"got {len(gene_symbols)} symbols. This indicates a bug in the mapping logic."
            )
        
        # Calculate statistics
        mapped_count = mapped_df.filter(pl.col('symbol').is_not_null()).height
        if has_feature_name:
            fallback_count = mapped_df.filter(
                pl.col('symbol').is_null() & 
                ~pl.col('feature_name').str.starts_with("ENS")
            ).height
        else:
            fallback_count = 0
        missing_count = n_genes - mapped_count - fallback_count
        
        action.log(
            message_type="gene_mapping_summary",
            total_genes=len(gene_symbols),
            mapped_via_hgnc=mapped_count,
            mapped_via_fallback=fallback_count,
            unmapped_using_ensembl_id=missing_count,
            hgnc_mapping_percentage=round(mapped_count / n_genes * 100, 2) if n_genes else 0,
            fallback_percentage=round(fallback_count / n_genes * 100, 2) if n_genes else 0,
            unmapped_percentage=round(missing_count / n_genes * 100, 2) if n_genes else 0
        )
        
        return gene_symbols


def create_cell_sentence(cell_expr: np.ndarray, gene_symbols: np.ndarray, top_n: int = 2000) -> str:
    """Create cell sentence from expression data.
    
    Uses argpartition for faster top-k selection (O(n) vs O(n log n) for full sort).
    Filters out Ensembl IDs to ensure only proper gene symbols are included.
    
    Args:
        cell_expr: Expression values for a single cell
        gene_symbols: Array of gene symbols corresponding to expression values
        top_n: Number of top genes to include in sentence
        
    Returns:
        Space-separated string of top expressed genes (excluding Ensembl IDs)
    """
    # First, filter out Ensembl IDs (genes starting with "ENS")
    # Create a mask for valid gene symbols (not Ensembl IDs)
    valid_mask = np.array([not str(gene).startswith("ENS") for gene in gene_symbols], dtype=bool)
    
    # Get indices of valid genes
    valid_indices = np.where(valid_mask)[0]
    
    # If we don't have enough valid genes, use what we have
    if len(valid_indices) == 0:
        # Fallback: no valid genes, return empty string or use all genes anyway
        # For now, let's return empty to make the problem visible
        return ""
    
    # Filter expression and symbols to only valid genes
    valid_expr = cell_expr[valid_indices]
    valid_symbols = gene_symbols[valid_indices]
    
    # Use argpartition for faster top-k selection (much faster than full sort)
    # Get indices of top N expressed genes from valid genes
    if top_n >= len(valid_expr):
        top_gene_indices = np.argsort(valid_expr)[::-1]
    else:
        # argpartition is O(n) vs argsort which is O(n log n)
        top_gene_indices = np.argpartition(valid_expr, -top_n)[-top_n:]
        # Sort only the top N (much smaller than sorting all)
        top_gene_indices = top_gene_indices[np.argsort(valid_expr[top_gene_indices])[::-1]]
    
    # Convert to gene symbols using numpy array indexing (faster than list comprehension)
    top_genes = valid_symbols[top_gene_indices]
    # Create space-separated string
    return ' '.join(top_genes.tolist())


def convert_h5ad_to_parquet(
    h5ad_path: Path,
    output_dir: Path = Path("./output"),
    chunk_size: int | None = None,
    target_mb: float | None = None,
    dataset_name: str | None = None,
    top_genes: int = 2000,
    compression: str = "zstd",
    compression_level: int = 3,
    use_pyarrow: bool = True
) -> None:
    """Convert h5ad file to parquet chunks with cell sentences.
    
    Args:
        h5ad_path: Path to the h5ad file
        output_dir: Directory to save parquet chunks
        chunk_size: Number of cells per chunk (used if target_mb is None). Default: None (uses target_mb)
        target_mb: Target size per chunk in MB (used if chunk_size is None). Default: None (uses chunk_size=2500)
        dataset_name: Name of the dataset for folder organization. If None, uses h5ad filename stem
        top_genes: Number of top expressed genes per cell
        compression: Compression algorithm for parquet files. Options: "uncompressed", "snappy", 
                     "gzip", "lzo", "brotli", "lz4", "zstd". Default: "zstd" (good balance)
        compression_level: Compression level (1-9 for zstd/gzip, 1-11 for brotli). Default: 3
        use_pyarrow: Use pyarrow backend for parquet writes (faster). Default: True
    """
    start_time = time.time()
    
    # Determine dataset name
    if dataset_name is None:
        dataset_name = h5ad_path.stem
    
    # Determine chunking strategy
    use_size_based = target_mb is not None
    if chunk_size is None and target_mb is None:
        # Default to row-based with 2500 rows (memory-friendly)
        chunk_size = 2500
        use_size_based = False
    elif chunk_size is not None and target_mb is not None:
        # Both specified, prefer size-based
        use_size_based = True
    
    with start_action(
        action_type="convert_h5ad_to_parquet",
        h5ad_path=str(h5ad_path),
        output_dir=str(output_dir),
        dataset_name=dataset_name,
        chunk_size=chunk_size,
        target_mb=target_mb,
        use_size_based=use_size_based,
        top_genes=top_genes
    ) as action:
        action.log(message_type="processing_started", h5ad_file=str(h5ad_path))
        
        # Load h5ad first to check if we need HGNC
        adata = load_h5ad_data(h5ad_path)
        
        # Determine if we need HGNC mapping
        # Use HGNC if:
        # AnnData doesn't have gene symbols in var_names AND feature_name doesn't have them either
        has_symbols = has_gene_symbols(adata)
        feature_has_symbols = feature_name_has_gene_symbols(adata)
        
        action.log(
            message_type="hgnc_usage_decision",
            has_gene_symbols=has_symbols,
            feature_name_has_gene_symbols=feature_has_symbols,
            has_feature_name='feature_name' in adata.var.columns,
            will_use_hgnc=True
        )
        
        # Load HGNC DataFrame (use Polars for fast joins!)
        # Use HGNC by default - download/create it automatically
        hgnc_df = None
        use_hgnc = True
        action.log(message_type="loading_hgnc_data_as_dataframe")
        try:
            # Load HGNC data from shared directory (parquet format)
            from cell2sentence4longevity.preprocessing.hgnc_mapper import get_hgnc_data
            shared_dir = Path("./data/shared")
            hgnc_df = get_hgnc_data(shared_dir)
            if hgnc_df is not None:
                action.log(message_type="hgnc_loaded_from_shared", gene_count=len(hgnc_df))
                use_hgnc = True
            else:
                action.log(message_type="hgnc_download_failed",
                          fallback="Will use var_names or feature_name from h5ad")
                use_hgnc = False
        except Exception as e:
            action.log(message_type="hgnc_loading_failed", error=str(e),
                      fallback="Will use var_names or feature_name from h5ad")
            use_hgnc = False
        
        # Map genes to symbols using Polars
        gene_symbols_list = map_genes_to_symbols_polars(adata, hgnc_df, use_hgnc=use_hgnc)
        # Convert to numpy array for faster indexing
        gene_symbols = np.array(gene_symbols_list, dtype=object)
        
        # Create output directory structure: output_dir/dataset_name/
        # If output_dir already ends with dataset_name, use it directly (avoid double nesting)
        if output_dir.name == dataset_name:
            chunks_dir = output_dir
        else:
            chunks_dir = output_dir / dataset_name
        chunks_dir.mkdir(parents=True, exist_ok=True)
        action.log(message_type="output_directory_created", chunks_dir=str(chunks_dir))
        
        # Get column names from obs once (outside loop)
        obs_columns = list(adata.obs.columns)
        
        # Helper: ensure Polars-compatible arrays (convert pandas Categorical to string)
        def _to_polars_compatible(values) -> np.ndarray:
            # Pandas categorical exposes dtype.name == 'category' even without importing pandas explicitly
            dtype_name = getattr(values, "dtype", None)
            dtype_name = getattr(dtype_name, "name", None)
            if dtype_name == "category":
                return values.astype(str).to_numpy()
            return values.to_numpy() if hasattr(values, "to_numpy") else np.asarray(values, dtype=object)
        
        n_cells = adata.n_obs
        chunk_idx = 0
        current_chunk_rows = []
        current_chunk_data = {}
        
        if use_size_based:
            target_bytes = int(target_mb * 1024 * 1024)
            # Use a batch size for processing (process in batches, then check size)
            batch_size = 5000  # Process 5000 cells at a time before checking size
            action.log(message_type="using_size_based_chunking", target_mb=target_mb, target_bytes=target_bytes, batch_size=batch_size)
            
            # Process in batches and accumulate until target size
            cell_idx = 0
            while cell_idx < n_cells:
                # Process a batch
                batch_end = min(cell_idx + batch_size, n_cells)
                batch_X = adata.X[cell_idx:batch_end].toarray()
                
                # Create cell sentences for batch
                batch_sentences = [
                    create_cell_sentence(cell_expr, gene_symbols, top_genes)
                    for cell_expr in batch_X
                ]
                
                # Get metadata for batch
                batch_obs_data = {}
                for col in obs_columns:
                    series_slice = adata.obs[col].iloc[cell_idx:batch_end]
                    batch_obs_data[col] = _to_polars_compatible(series_slice)
                batch_obs_data['cell_sentence'] = batch_sentences
                
                # Add batch to current chunk
                if not current_chunk_data:
                    # Initialize with batch data
                    current_chunk_data = batch_obs_data
                else:
                    # Append batch to existing chunk
                    for col in batch_obs_data:
                        if isinstance(current_chunk_data[col], list):
                            current_chunk_data[col].extend(batch_obs_data[col])
                        else:
                            # Convert to list and extend
                            current_chunk_data[col] = list(current_chunk_data[col]) + list(batch_obs_data[col])
                
                # Check size by creating DataFrame and writing to temp file
                chunk_df = pl.DataFrame(current_chunk_data)
                # Add age column using Polars expression
                chunk_df = extract_age_column(chunk_df)
                with tempfile.NamedTemporaryFile(delete=False, suffix='.parquet') as tmp_file:
                    temp_path = Path(tmp_file.name)
                    chunk_df.write_parquet(
                        temp_path,
                        compression=compression,
                        compression_level=compression_level,
                        use_pyarrow=use_pyarrow,
                    )
                    file_size = temp_path.stat().st_size
                    os.unlink(temp_path)
                
                # Write chunk if we've reached target size or at end
                if file_size >= target_bytes or batch_end == n_cells:
                    output_file = chunks_dir / f"chunk_{chunk_idx:04d}.parquet"
                    chunk_df.write_parquet(
                        output_file,
                        compression=compression,
                        compression_level=compression_level,
                        use_pyarrow=use_pyarrow,
                    )
                    
                    file_size_mb = output_file.stat().st_size / (1024 * 1024)
                    action.log(
                        message_type="chunk_saved",
                        chunk_idx=chunk_idx,
                        output_file=str(output_file),
                        rows=len(chunk_df),
                        size_mb=round(file_size_mb, 2)
                    )
                    
                    # Reset for next chunk
                    chunk_idx += 1
                    current_chunk_data = {}
                
                cell_idx = batch_end
            
            n_chunks = chunk_idx
        else:
            # Row-based chunking (original logic)
            n_chunks = (n_cells + chunk_size - 1) // chunk_size
            action.log(message_type="using_row_based_chunking", chunk_size=chunk_size, n_chunks=n_chunks)
            
            for chunk_idx in tqdm(range(n_chunks), desc='Processing chunks'):
                with start_action(action_type="process_chunk", chunk_idx=chunk_idx) as chunk_action:
                    start_idx = chunk_idx * chunk_size
                    end_idx = min(start_idx + chunk_size, n_cells)
                    
                    # Read chunk of expression matrix
                    chunk_X = adata.X[start_idx:end_idx].toarray()
                    
                    # Create cell sentences for each cell in chunk
                    # Using list comprehension is faster than explicit loop
                    cell_sentences = [
                        create_cell_sentence(cell_expr, gene_symbols, top_genes)
                        for cell_expr in chunk_X
                    ]
                    
                    # Build Polars DataFrame directly from obs data (skip pandas intermediate)
                    # Get metadata for this chunk
                    chunk_obs_data = {}
                    for col in obs_columns:
                        series_slice = adata.obs[col].iloc[start_idx:end_idx]
                        chunk_obs_data[col] = _to_polars_compatible(series_slice)
                    
                    # Add cell_sentence column
                    chunk_obs_data['cell_sentence'] = cell_sentences
                    
                    # Create Polars DataFrame directly
                    chunk_df = pl.DataFrame(chunk_obs_data)
                    # Add age column using Polars expression
                    chunk_df = extract_age_column(chunk_df)
                    
                    output_file = chunks_dir / f"chunk_{chunk_idx:04d}.parquet"
                    # Use compression and other optimizations for faster writes
                    chunk_df.write_parquet(
                        output_file,
                        compression=compression,
                        compression_level=compression_level,
                        use_pyarrow=use_pyarrow,
                    )
                    
                    file_size_mb = output_file.stat().st_size / (1024 * 1024)
                    chunk_action.log(
                        message_type="chunk_saved",
                        output_file=str(output_file),
                        rows=len(chunk_df),
                        size_mb=round(file_size_mb, 2)
                    )
                    
                    # Clear memory
                    del chunk_X, cell_sentences, chunk_obs_data, chunk_df
        
        adata.file.close()
        
        # Calculate and log total processing time
        end_time = time.time()
        processing_time_seconds = end_time - start_time
        
        # Format time as hh:mm:ss
        hours = int(processing_time_seconds // 3600)
        minutes = int((processing_time_seconds % 3600) // 60)
        seconds = int(processing_time_seconds % 60)
        processing_time_formatted = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        
        action.log(
            message_type="conversion_complete",
            n_chunks=n_chunks,
            processing_time_seconds=round(processing_time_seconds, 2),
            processing_time_formatted=processing_time_formatted,
            h5ad_file=str(h5ad_path),
            chunks_dir=str(chunks_dir)
        )


def convert_h5ad_to_train_test(
    h5ad_path: Path,
    output_dir: Path = Path("./output"),
    dataset_name: str | None = None,
    chunk_size: int = 10000,
    top_genes: int = 2000,
    test_size: float = 0.05,
    random_state: int = 42,
    compression: str = "zstd",
    compression_level: int = 3,
    use_pyarrow: bool = True,
    skip_train_test_split: bool = False,
    stratify_by_age: bool = True,
    join_collection: bool = True
) -> None:
    """Convert h5ad file directly to train/test parquet splits in one streaming pass.
    
    This unified function:
    1. Reads h5ad chunks
    2. Creates cell sentences
    3. Extracts age from development_stage
    4. Adds dataset_id column for joining with collection metadata (if applicable)
    5. Assigns cells to train/test split (stratified by age using pure Polars)
    6. Writes directly to train/test output directories
    
    No interim files, no pandas/sklearn conversion - pure Polars for memory efficiency.
    
    Collection joining behavior:
    - By default (join_collection=True), auto-detects if dataset is from CellxGene (by UUID pattern or URL)
    - If detected, always adds dataset_id column (regardless of whether dataset is in collections cache)
    - Only joins with collections metadata if dataset is found in collections cache
    - If join_collection=False, skips collection joining entirely and does not add dataset_id column
    
    Args:
        h5ad_path: Path to the h5ad file
        output_dir: Directory to save train/test splits
        dataset_name: Name of the dataset for folder organization. If None, uses h5ad filename stem
        chunk_size: Number of cells per chunk
        top_genes: Number of top expressed genes per cell
        test_size: Proportion of data for test set (0.0 to 1.0)
        random_state: Random seed for reproducibility
        compression: Compression algorithm for parquet files
        compression_level: Compression level
        use_pyarrow: Use pyarrow backend for parquet writes
        skip_train_test_split: If True, writes all data to output_dir without splitting
        stratify_by_age: If True, maintains age distribution in train/test splits (default: True)
        join_collection: If True (default), auto-detects cellxgene datasets and always adds dataset_id column. Only joins with collections metadata if dataset is found in collections cache. If False, skips collection joining and does not add dataset_id column.
    """
    start_time = time.time()
    
    # Determine dataset name
    if dataset_name is None:
        dataset_name = h5ad_path.stem
    
    with start_action(
        action_type="convert_h5ad_to_train_test",
        h5ad_path=str(h5ad_path),
        output_dir=str(output_dir),
        dataset_name=dataset_name,
        chunk_size=chunk_size,
        top_genes=top_genes,
        test_size=test_size,
        skip_train_test_split=skip_train_test_split,
        stratify_by_age=stratify_by_age,
        join_collection=join_collection
    ) as action:
        action.log(message_type="processing_started", h5ad_file=str(h5ad_path))
        
        # Load h5ad first to check if we need HGNC
        adata = load_h5ad_data(h5ad_path)
        
        # Determine if we need HGNC mapping
        has_symbols = has_gene_symbols(adata)
        feature_has_symbols = feature_name_has_gene_symbols(adata)
        
        action.log(
            message_type="hgnc_usage_decision",
            has_gene_symbols=has_symbols,
            feature_name_has_gene_symbols=feature_has_symbols,
            has_feature_name='feature_name' in adata.var.columns,
            will_use_hgnc=True
        )
        
        # Load HGNC DataFrame (use Polars for fast joins!)
        # Use HGNC by default - download/create it automatically
        hgnc_df = None
        use_hgnc = True
        action.log(message_type="loading_hgnc_data_as_dataframe")
        try:
            # Load HGNC data from shared directory (parquet format)
            from cell2sentence4longevity.preprocessing.hgnc_mapper import get_hgnc_data
            shared_dir = Path("./data/shared")
            hgnc_df = get_hgnc_data(shared_dir)
            if hgnc_df is not None:
                action.log(message_type="hgnc_loaded_from_shared", gene_count=len(hgnc_df))
                use_hgnc = True
            else:
                action.log(message_type="hgnc_download_failed",
                          fallback="Will use var_names or feature_name from h5ad")
                use_hgnc = False
        except Exception as e:
            action.log(message_type="hgnc_loading_failed", error=str(e),
                      fallback="Will use var_names or feature_name from h5ad")
            use_hgnc = False
        
        # Map genes to symbols using Polars
        gene_symbols_list = map_genes_to_symbols_polars(adata, hgnc_df, use_hgnc=use_hgnc)
        gene_symbols = np.array(gene_symbols_list, dtype=object)
        
        # Create output directory structure
        if skip_train_test_split:
            # Single output directory
            output_chunks_dir = output_dir / dataset_name
            dir_existed = output_chunks_dir.exists()
            output_chunks_dir.mkdir(parents=True, exist_ok=True)
            action.log(message_type="output_directory_created" if not dir_existed else "output_directory_exists", 
                      output_dir=str(output_chunks_dir),
                      mode="single_directory",
                      created=not dir_existed)
        else:
            # Train/test split directories
            train_dir = output_dir / dataset_name / "train"
            test_dir = output_dir / dataset_name / "test"
            train_existed = train_dir.exists()
            test_existed = test_dir.exists()
            train_dir.mkdir(parents=True, exist_ok=True)
            test_dir.mkdir(parents=True, exist_ok=True)
            action.log(message_type="output_directories_created" if (not train_existed or not test_existed) else "output_directories_exist",
                      train_dir=str(train_dir),
                      test_dir=str(test_dir),
                      mode="train_test_split",
                      train_created=not train_existed,
                      test_created=not test_existed)
        
        # Determine if we should add dataset_id column and join with collections
        # By default (join_collection=True), auto-detect cellxgene datasets and check if they exist in collections
        # If join_collection=False, skip joining entirely
        should_add_dataset_id = False
        should_join_collections = False
        dataset_id = None
        
        if join_collection:
            # Auto-detect: check if it's a cellxgene dataset
            from cell2sentence4longevity.preprocessing.publication_lookup import (
                is_cellxgene_dataset,
                extract_dataset_id_from_path,
                dataset_id_exists_in_collections,
                join_with_collections
            )
            if is_cellxgene_dataset(h5ad_path):
                dataset_id = extract_dataset_id_from_path(h5ad_path)
                # Always add dataset_id column for cellxgene datasets
                should_add_dataset_id = True
                
                # Only join with collections if dataset exists in collections cache
                found_in_collections = dataset_id_exists_in_collections(dataset_id, cache_dir=None)
                if found_in_collections:
                    should_join_collections = True
                    action.log(
                        message_type="dataset_id_found_in_collections",
                        dataset_id=dataset_id,
                        found_in_collections=True,
                        will_add_dataset_id=True,
                        will_join_collections=True,
                        note="Auto-detected cellxgene dataset found in collections, will add dataset_id column and join with collections"
                    )
                else:
                    action.log(
                        message_type="dataset_id_not_found_in_collections",
                        dataset_id=dataset_id,
                        found_in_collections=False,
                        will_add_dataset_id=True,
                        will_join_collections=False,
                        note="Auto-detected cellxgene dataset but not found in collections, will add dataset_id column but skip collection join"
                    )
            else:
                action.log(
                    message_type="not_cellxgene_dataset",
                    found_in_collections=False,
                    will_add_dataset_id=False,
                    will_join_collections=False,
                    note="Dataset does not appear to be from cellxgene, skipping dataset_id column"
                )
        else:
            action.log(
                message_type="join_collection_disabled",
                found_in_collections=False,
                will_add_dataset_id=False,
                will_join_collections=False,
                note="join_collection=False, skipping collection joining and dataset_id column"
            )
        
        # Get column names from obs once (outside loop)
        obs_columns = list(adata.obs.columns)
        
        # Helper: ensure Polars-compatible arrays
        def _to_polars_compatible(values) -> np.ndarray:
            dtype_name = getattr(values, "dtype", None)
            dtype_name = getattr(dtype_name, "name", None)
            if dtype_name == "category":
                return values.astype(str).to_numpy()
            return values.to_numpy() if hasattr(values, "to_numpy") else np.asarray(values, dtype=object)
        
        n_cells = adata.n_obs
        n_chunks = (n_cells + chunk_size - 1) // chunk_size
        
        # Initialize counters for train/test
        train_chunk_idx = 0
        test_chunk_idx = 0
        train_buffer = []
        test_buffer = []
        train_buffer_data = {}
        test_buffer_data = {}
        
        # Setup random number generator for reproducible splits
        rng = np.random.RandomState(random_state)
        
        # Statistics
        total_cells_processed = 0
        cells_with_null_age = 0
        cells_filtered_out = 0
        train_cells = 0
        test_cells = 0
        
        action.log(message_type="starting_chunk_processing", 
                  total_chunks=n_chunks,
                  cells_per_chunk=chunk_size)
        
        for chunk_idx in tqdm(range(n_chunks), desc='Processing chunks'):
            with start_action(action_type="process_chunk", chunk_idx=chunk_idx) as chunk_action:
                start_idx = chunk_idx * chunk_size
                end_idx = min(start_idx + chunk_size, n_cells)
                
                # Read chunk of expression matrix
                chunk_X = adata.X[start_idx:end_idx].toarray()
                
                # Create cell sentences for each cell in chunk
                cell_sentences = [
                    create_cell_sentence(cell_expr, gene_symbols, top_genes)
                    for cell_expr in chunk_X
                ]
                
                # Build chunk data with all metadata
                chunk_obs_data = {}
                for col in obs_columns:
                    series_slice = adata.obs[col].iloc[start_idx:end_idx]
                    chunk_obs_data[col] = _to_polars_compatible(series_slice)
                
                # Add cell_sentence column
                chunk_obs_data['cell_sentence'] = cell_sentences
                
                # Create Polars DataFrame
                chunk_df = pl.DataFrame(chunk_obs_data)
                
                # Extract age using vectorized Polars regex (much faster than map_elements)
                chunk_df = extract_age_column(chunk_df)
                
                # Add dataset_id column if we determined it should be added
                if should_add_dataset_id and dataset_id is not None:
                    chunk_df = chunk_df.with_columns([
                        pl.lit(dataset_id).alias("dataset_id")
                    ])
                
                # Join with collections to add publication metadata
                if should_join_collections:
                    chunk_df = join_with_collections(chunk_df, cache_dir=None)
                
                if 'development_stage' not in chunk_df.columns:
                    chunk_action.log(message_type="warning_no_development_stage",
                                   warning="No development_stage column found, age will be null")
                
                # Count null ages in this chunk
                null_ages_in_chunk = chunk_df.filter(pl.col('age').is_null()).height
                cells_with_null_age += null_ages_in_chunk
                
                # Filter out null ages
                chunk_df = chunk_df.filter(pl.col('age').is_not_null())
                cells_filtered_out += null_ages_in_chunk
                total_cells_processed += (end_idx - start_idx)
                
                if chunk_df.height == 0:
                    chunk_action.log(message_type="chunk_empty_after_filtering",
                                   warning="All cells filtered out due to null age")
                    continue
                
                chunk_action.log(message_type="chunk_processed",
                               cells_in=end_idx - start_idx,
                               cells_after_age_filter=chunk_df.height,
                               cells_filtered=null_ages_in_chunk)
                
                # Split into train/test or write to single directory
                if skip_train_test_split:
                    # Write entire chunk to single output directory
                    output_file = output_chunks_dir / f"chunk_{chunk_idx:04d}.parquet"
                    chunk_df.write_parquet(
                        output_file,
                        compression=compression,
                        compression_level=compression_level,
                        use_pyarrow=use_pyarrow,
                    )
                    chunk_action.log(message_type="chunk_saved",
                                   output_file=str(output_file),
                                   rows=len(chunk_df))
                else:
                    # Stratified or random split using pure Polars
                    if stratify_by_age:
                        # Stratified split: maintain age distribution across train/test
                        # Add random number for stratified sampling within each age group
                        chunk_df = chunk_df.with_columns(
                            pl.lit(rng.random(chunk_df.height)).alias('_random_split')
                        )
                        
                        # For each age group, split proportionally
                        # Group by age, then filter by random number threshold
                        train_chunk = (
                            chunk_df
                            .group_by('age', maintain_order=True)
                            .agg(pl.all())
                            .explode(pl.all().exclude('age'))
                            .filter(pl.col('_random_split') >= test_size)
                            .drop('_random_split')
                        )
                        
                        test_chunk = (
                            chunk_df
                            .group_by('age', maintain_order=True)
                            .agg(pl.all())
                            .explode(pl.all().exclude('age'))
                            .filter(pl.col('_random_split') < test_size)
                            .drop('_random_split')
                        )
                    else:
                        # Simple random split (faster, no stratification)
                        split_assignments = rng.random(chunk_df.height) >= test_size
                        train_chunk = chunk_df.filter(pl.Series(split_assignments))
                        test_chunk = chunk_df.filter(~pl.Series(split_assignments))
                    
                    train_cells += train_chunk.height
                    test_cells += test_chunk.height
                    
                    # Add to buffers
                    if train_chunk.height > 0:
                        train_buffer.append(train_chunk)
                        
                        # Check if we should flush train buffer
                        total_buffered = sum(df.height for df in train_buffer)
                        if total_buffered >= chunk_size:
                            # Concatenate and write
                            train_df_combined = pl.concat(train_buffer)
                            output_file = train_dir / f"chunk_{train_chunk_idx:04d}.parquet"
                            train_df_combined.write_parquet(
                                output_file,
                                compression=compression,
                                compression_level=compression_level,
                                use_pyarrow=use_pyarrow,
                            )
                            chunk_action.log(message_type="train_chunk_saved",
                                           output_file=str(output_file),
                                           rows=len(train_df_combined))
                            train_chunk_idx += 1
                            train_buffer = []
                    
                    if test_chunk.height > 0:
                        test_buffer.append(test_chunk)
                        
                        # Check if we should flush test buffer
                        total_buffered = sum(df.height for df in test_buffer)
                        if total_buffered >= chunk_size:
                            # Concatenate and write
                            test_df_combined = pl.concat(test_buffer)
                            output_file = test_dir / f"chunk_{test_chunk_idx:04d}.parquet"
                            test_df_combined.write_parquet(
                                output_file,
                                compression=compression,
                                compression_level=compression_level,
                                use_pyarrow=use_pyarrow,
                            )
                            chunk_action.log(message_type="test_chunk_saved",
                                           output_file=str(output_file),
                                           rows=len(test_df_combined))
                            test_chunk_idx += 1
                            test_buffer = []
                
                # Clear memory
                del chunk_X, cell_sentences, chunk_obs_data, chunk_df
        
        # Flush remaining buffers
        if not skip_train_test_split:
            if train_buffer:
                train_df_combined = pl.concat(train_buffer)
                output_file = train_dir / f"chunk_{train_chunk_idx:04d}.parquet"
                train_df_combined.write_parquet(
                    output_file,
                    compression=compression,
                    compression_level=compression_level,
                    use_pyarrow=use_pyarrow,
                )
                action.log(message_type="final_train_chunk_saved",
                          output_file=str(output_file),
                          rows=len(train_df_combined))
                train_chunk_idx += 1
            
            if test_buffer:
                test_df_combined = pl.concat(test_buffer)
                output_file = test_dir / f"chunk_{test_chunk_idx:04d}.parquet"
                test_df_combined.write_parquet(
                    output_file,
                    compression=compression,
                    compression_level=compression_level,
                    use_pyarrow=use_pyarrow,
                )
                action.log(message_type="final_test_chunk_saved",
                          output_file=str(output_file),
                          rows=len(test_df_combined))
                test_chunk_idx += 1
        
        adata.file.close()
        
        # Calculate and log total processing time
        end_time = time.time()
        processing_time_seconds = end_time - start_time
        
        # Format time as hh:mm:ss
        hours = int(processing_time_seconds // 3600)
        minutes = int((processing_time_seconds % 3600) // 60)
        seconds = int(processing_time_seconds % 60)
        processing_time_formatted = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        
        # Log statistics
        if skip_train_test_split:
            action.log(
                message_type="conversion_complete",
                total_cells_processed=total_cells_processed,
                cells_with_null_age=cells_with_null_age,
                cells_filtered_out=cells_filtered_out,
                cells_written=total_cells_processed - cells_filtered_out,
                processing_time_seconds=round(processing_time_seconds, 2),
                processing_time_formatted=processing_time_formatted,
                h5ad_file=str(h5ad_path),
                output_dir=str(output_chunks_dir)
            )
        else:
            actual_test_ratio = test_cells / (train_cells + test_cells) if (train_cells + test_cells) > 0 else 0
            action.log(
                message_type="conversion_complete_with_split",
                total_cells_processed=total_cells_processed,
                cells_with_null_age=cells_with_null_age,
                cells_filtered_out=cells_filtered_out,
                train_cells=train_cells,
                test_cells=test_cells,
                train_chunks=train_chunk_idx,
                test_chunks=test_chunk_idx,
                actual_test_ratio=round(actual_test_ratio, 4),
                target_test_ratio=test_size,
                stratified=stratify_by_age,
                split_method="stratified_by_age" if stratify_by_age else "random",
                processing_time_seconds=round(processing_time_seconds, 2),
                processing_time_formatted=processing_time_formatted,
                h5ad_file=str(h5ad_path),
                train_dir=str(train_dir),
                test_dir=str(test_dir)
            )

