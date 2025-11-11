"""H5AD to Parquet converter with cell sentence generation."""

from pathlib import Path
from typing import Dict, List
import pickle
import time
import tempfile
import os

import anndata as ad
import numpy as np
from eliot import start_action
from tqdm import tqdm
import polars as pl


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


def map_genes_to_symbols(
    adata: ad.AnnData, 
    ensembl_to_symbol: Dict[str, str]
) -> List[str]:
    """Map Ensembl IDs to gene symbols.
    
    Args:
        adata: AnnData object
        ensembl_to_symbol: Mapping dictionary from Ensembl ID to gene symbol
        
    Returns:
        List of gene symbols
    """
    with start_action(action_type="map_genes_to_symbols") as action:
        # Access var_names without converting to list to save memory
        n_genes = adata.n_vars
        gene_symbols = []
        mapped_count = 0
        fallback_count = 0
        missing_count = 0
        
        action.log(message_type="starting_gene_mapping", total_genes=n_genes)
        
        # Iterate through var_names directly instead of creating a list
        for i, ens_id in enumerate(adata.var_names):
            if ens_id in ensembl_to_symbol:
                symbol = ensembl_to_symbol[ens_id]
                gene_symbols.append(symbol)
                mapped_count += 1
            else:
                # Fallback to feature_name if HGNC doesn't have mapping
                if 'feature_name' in adata.var.columns:
                    symbol = adata.var['feature_name'].iloc[i]
                    gene_symbols.append(symbol)
                    fallback_count += 1
                else:
                    # No feature_name available, use ensembl ID directly
                    gene_symbols.append(ens_id)
                    missing_count += 1
        
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
        
        if missing_count > 0:
            # Log sample unmapped genes for debugging (only first 10)
            unmapped_samples = []
            count = 0
            for ens_id in adata.var_names:
                if ens_id not in ensembl_to_symbol and 'feature_name' not in adata.var.columns:
                    unmapped_samples.append(ens_id)
                    count += 1
                    if count >= 10:
                        break
            action.log(message_type="sample_unmapped_genes", samples=unmapped_samples)
        
        return gene_symbols


def create_cell_sentence(cell_expr: np.ndarray, gene_symbols: np.ndarray, top_n: int = 2000) -> str:
    """Create cell sentence from expression data.
    
    Uses argpartition for faster top-k selection (O(n) vs O(n log n) for full sort).
    
    Args:
        cell_expr: Expression values for a single cell
        gene_symbols: Array of gene symbols corresponding to expression values
        top_n: Number of top genes to include in sentence
        
    Returns:
        Space-separated string of top expressed genes
    """
    # Use argpartition for faster top-k selection (much faster than full sort)
    # Get indices of top N expressed genes
    if top_n >= len(cell_expr):
        top_gene_indices = np.argsort(cell_expr)[::-1]
    else:
        # argpartition is O(n) vs argsort which is O(n log n)
        top_gene_indices = np.argpartition(cell_expr, -top_n)[-top_n:]
        # Sort only the top N (much smaller than sorting all)
        top_gene_indices = top_gene_indices[np.argsort(cell_expr[top_gene_indices])[::-1]]
    
    # Convert to gene symbols using numpy array indexing (faster than list comprehension)
    top_genes = gene_symbols[top_gene_indices]
    # Create space-separated string
    return ' '.join(top_genes.tolist())


def convert_h5ad_to_parquet(
    h5ad_path: Path,
    mappers_path: Path | None = None,
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
        mappers_path: Path to the HGNC mappers pickle file (optional)
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
        
        # Load mappers if available
        ensembl_to_symbol = {}
        if mappers_path and mappers_path.exists():
            action.log(message_type="loading_mappers", path=str(mappers_path))
            try:
                with open(mappers_path, 'rb') as f:
                    mappers = pickle.load(f)
                ensembl_to_symbol = mappers['ensembl_to_symbol']
                action.log(message_type="mappers_loaded", mapper_count=len(ensembl_to_symbol))
            except Exception as e:
                action.log(message_type="mappers_load_failed", error=str(e), 
                          fallback="Will use feature_name from h5ad")
        else:
            action.log(message_type="no_mappers_provided", 
                      fallback="Will use feature_name from h5ad if available")
        
        # Load h5ad
        adata = load_h5ad_data(h5ad_path)
        
        # Map genes to symbols
        gene_symbols_list = map_genes_to_symbols(adata, ensembl_to_symbol)
        # Convert to numpy array for faster indexing
        gene_symbols = np.array(gene_symbols_list, dtype=object)
        
        # Create output directory structure: output_dir/dataset_name/chunks/
        chunks_dir = output_dir / dataset_name / "chunks"
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

