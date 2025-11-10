"""H5AD to Parquet converter with cell sentence generation."""

from pathlib import Path
from typing import Dict, List
import pickle

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
        
        # Log available metadata columns
        obs_columns = list(adata.obs.columns)
        var_columns = list(adata.var.columns)
        
        action.log(
            message_type="metadata_structure",
            n_obs_columns=len(obs_columns),
            n_var_columns=len(var_columns),
            obs_columns=obs_columns,
            var_columns=var_columns
        )
        
        # Check for critical columns
        has_development_stage = 'development_stage' in obs_columns
        has_feature_name = 'feature_name' in var_columns
        
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
        ensembl_ids = list(adata.var_names)
        gene_symbols = []
        mapped_count = 0
        fallback_count = 0
        missing_count = 0
        
        action.log(message_type="starting_gene_mapping", total_genes=len(ensembl_ids))
        
        for i, ens_id in enumerate(ensembl_ids):
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
            hgnc_mapping_percentage=round(mapped_count / len(ensembl_ids) * 100, 2) if ensembl_ids else 0,
            fallback_percentage=round(fallback_count / len(ensembl_ids) * 100, 2) if ensembl_ids else 0,
            unmapped_percentage=round(missing_count / len(ensembl_ids) * 100, 2) if ensembl_ids else 0
        )
        
        if missing_count > 0:
            # Log sample unmapped genes for debugging
            unmapped_samples = [ens_id for i, ens_id in enumerate(ensembl_ids) 
                              if ens_id not in ensembl_to_symbol and 
                              'feature_name' not in adata.var.columns][:10]
            action.log(message_type="sample_unmapped_genes", samples=unmapped_samples)
        
        return gene_symbols


def create_cell_sentence(cell_expr: np.ndarray, gene_symbols: List[str], top_n: int = 2000) -> str:
    """Create cell sentence from expression data.
    
    Args:
        cell_expr: Expression values for a single cell
        gene_symbols: List of gene symbols corresponding to expression values
        top_n: Number of top genes to include in sentence
        
    Returns:
        Space-separated string of top expressed genes
    """
    # Get indices of top N expressed genes
    top_gene_indices = np.argsort(cell_expr)[::-1][:top_n]
    # Convert to gene symbols
    top_genes = [gene_symbols[idx] for idx in top_gene_indices]
    # Create space-separated string
    return ' '.join(top_genes)


def convert_h5ad_to_parquet(
    h5ad_path: Path,
    mappers_path: Path,
    output_dir: Path,
    chunk_size: int = 10000,
    top_genes: int = 2000
) -> None:
    """Convert h5ad file to parquet chunks with cell sentences.
    
    Args:
        h5ad_path: Path to the h5ad file
        mappers_path: Path to the HGNC mappers pickle file
        output_dir: Directory to save parquet chunks
        chunk_size: Number of cells per chunk
        top_genes: Number of top expressed genes per cell
    """
    with start_action(
        action_type="convert_h5ad_to_parquet",
        h5ad_path=str(h5ad_path),
        output_dir=str(output_dir),
        chunk_size=chunk_size,
        top_genes=top_genes
    ) as action:
        # Load mappers
        action.log(message_type="loading_mappers")
        with open(mappers_path, 'rb') as f:
            mappers = pickle.load(f)
        ensembl_to_symbol = mappers['ensembl_to_symbol']
        action.log(message_type="mappers_loaded", mapper_count=len(ensembl_to_symbol))
        
        # Load h5ad
        adata = load_h5ad_data(h5ad_path)
        
        # Map genes to symbols
        gene_symbols = map_genes_to_symbols(adata, ensembl_to_symbol)
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Process in chunks
        n_cells = adata.n_obs
        n_chunks = (n_cells + chunk_size - 1) // chunk_size
        action.log(message_type="processing_chunks", n_chunks=n_chunks)
        
        for chunk_idx in tqdm(range(n_chunks), desc='Processing chunks'):
            with start_action(action_type="process_chunk", chunk_idx=chunk_idx) as chunk_action:
                start_idx = chunk_idx * chunk_size
                end_idx = min(start_idx + chunk_size, n_cells)
                
                # Read chunk of expression matrix
                chunk_X = adata.X[start_idx:end_idx].toarray()
                
                # Get metadata for this chunk - convert to dict first
                chunk_obs_pd = adata.obs.iloc[start_idx:end_idx].copy()
                
                # Create cell sentences for each cell in chunk
                cell_sentences = []
                for cell_expr in chunk_X:
                    cell_sentence = create_cell_sentence(cell_expr, gene_symbols, top_genes)
                    cell_sentences.append(cell_sentence)
                
                # Add cell_sentence column
                chunk_obs_pd['cell_sentence'] = cell_sentences
                
                # Convert to Polars and save
                chunk_df = pl.from_pandas(chunk_obs_pd)
                output_file = output_dir / f"chunk_{chunk_idx:04d}.parquet"
                chunk_df.write_parquet(output_file)
                
                chunk_action.log(message_type="chunk_saved", output_file=str(output_file))
                
                # Clear memory
                del chunk_X, cell_sentences, chunk_obs_pd, chunk_df
        
        adata.file.close()
        action.log(message_type="conversion_complete", n_chunks=n_chunks)

