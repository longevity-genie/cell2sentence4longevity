"""HGNC gene mapper creation module."""

from pathlib import Path
import io

import requests
import polars as pl
from eliot import start_action

# Default shared directory for HGNC data
DEFAULT_SHARED_DIR = Path("./data/shared")
HGNC_TSV_NAME = "hgnc_complete_set.parquet"


def download_hgnc_data(shared_dir: Path | None = None) -> pl.DataFrame | None:
    """Download HGNC complete dataset and save to shared directory as parquet.
    
    Args:
        shared_dir: Directory to save the downloaded data. Defaults to DEFAULT_SHARED_DIR.
        
    Returns:
        DataFrame with HGNC data or None if download fails
    """
    if shared_dir is None:
        shared_dir = DEFAULT_SHARED_DIR
    
    shared_dir.mkdir(parents=True, exist_ok=True)
    
    with start_action(action_type="download_hgnc_data") as action:
        url = "https://storage.googleapis.com/public-download-files/hgnc/tsv/tsv/hgnc_complete_set.txt"
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; cell2sentence4longevity/0.1)"
        }
        output_parquet = shared_dir / HGNC_TSV_NAME
        
        action.log(message_type="attempt_download", url=url)
        
        try:
            response = requests.get(url, timeout=120, headers=headers)
            response.raise_for_status()
            
            # Use Polars to read the TSV data with schema overrides
            # omim_id and other columns can have pipe-separated values like "312095|465000"
            schema_overrides = {
                'omim_id': pl.String,
                'ena': pl.String,
                'refseq_accession': pl.String,
                'ccds_id': pl.String,
                'uniprot_ids': pl.String,
                'pubmed_id': pl.String,
                'mgd_id': pl.String,
                'rgd_id': pl.String
            }
            hgnc_df = pl.read_csv(
                io.StringIO(response.text), 
                separator='\t',
                schema_overrides=schema_overrides,
                infer_schema_length=10000
            )
            action.log(message_type="download_success", gene_count=len(hgnc_df))
            
            # Save to parquet with compression
            hgnc_df.write_parquet(output_parquet, compression="zstd", compression_level=3)
            action.log(message_type="saved_to_file", path=str(output_parquet))
            
            return hgnc_df
            
        except Exception as e:
            action.log(message_type="download_failed", error=str(e))
            
            # Check if local parquet file exists as fallback
            action.log(message_type="checking_local_file")
            if output_parquet.exists():
                hgnc_df = pl.read_parquet(output_parquet)
                action.log(message_type="loaded_from_local", gene_count=len(hgnc_df))
                return hgnc_df
            
            # Also check for old TSV format for backward compatibility
            old_tsv = shared_dir / "hgnc_complete_set.tsv"
            if old_tsv.exists():
                action.log(message_type="loading_from_old_tsv_format")
                schema_overrides = {
                    'omim_id': pl.String,
                    'ena': pl.String,
                    'refseq_accession': pl.String,
                    'ccds_id': pl.String,
                    'uniprot_ids': pl.String,
                    'pubmed_id': pl.String,
                    'mgd_id': pl.String,
                    'rgd_id': pl.String
                }
                hgnc_df = pl.read_csv(
                    old_tsv, 
                    separator='\t',
                    schema_overrides=schema_overrides,
                    infer_schema_length=10000
                )
                # Convert to parquet for future use
                hgnc_df.write_parquet(output_parquet, compression="zstd", compression_level=3)
                action.log(message_type="converted_tsv_to_parquet", gene_count=len(hgnc_df))
                return hgnc_df
            
            action.log(message_type="download_failed_no_local")
            return None


def get_hgnc_data(shared_dir: Path | None = None) -> pl.DataFrame | None:
    """Get HGNC data, loading from cache or downloading if needed.
    
    Args:
        shared_dir: Directory to look for cached data. Defaults to DEFAULT_SHARED_DIR.
        
    Returns:
        DataFrame with HGNC data or None if unavailable
    """
    if shared_dir is None:
        shared_dir = DEFAULT_SHARED_DIR
    
    parquet_path = shared_dir / HGNC_TSV_NAME
    
    if parquet_path.exists():
        return pl.read_parquet(parquet_path)
    
    # Try to download
    return download_hgnc_data(shared_dir)

