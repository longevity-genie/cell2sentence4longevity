"""HGNC gene mapper creation module."""

from pathlib import Path
from typing import Dict
import io
import pickle

import requests
import polars as pl
from eliot import start_action

# Default shared directory for HGNC data
DEFAULT_SHARED_DIR = Path("./data/shared")
HGNC_TSV_NAME = "hgnc_complete_set.parquet"
HGNC_MAPPERS_NAME = "hgnc_mappers.parquet"


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


def create_mappers(hgnc_df: pl.DataFrame) -> Dict[str, Dict[str, str]]:
    """Create mapping dictionaries from HGNC data.
    
    Args:
        hgnc_df: HGNC DataFrame
        
    Returns:
        Dictionary containing various gene mappers
    """
    with start_action(action_type="create_mappers") as action:
        mappers: Dict[str, Dict[str, str]] = {}
        
        # Normalize column names (handle different HGNC formats)
        col_map = {
            'Ensembl gene ID': 'ensembl_gene_id',
            'Approved symbol': 'symbol',
            'Previous symbols': 'prev_symbol',
            'Alias symbols': 'alias_symbol'
        }
        
        for old_name, new_name in col_map.items():
            if old_name in hgnc_df.columns:
                hgnc_df = hgnc_df.rename({old_name: new_name})
        
        # 1. Ensembl ID -> Official Symbol (vectorized - much faster than iter_rows)
        action.log(message_type="creating_mapper", mapper_type="ensembl_to_symbol")
        filtered_df = hgnc_df.filter(
            pl.col('ensembl_gene_id').is_not_null() & pl.col('symbol').is_not_null()
        ).select(['ensembl_gene_id', 'symbol'])
        # Use to_dict with as_series=False to get dict of lists, then convert to dict
        ensembl_to_symbol = dict(zip(
            filtered_df['ensembl_gene_id'].to_list(),
            filtered_df['symbol'].to_list()
        ))
        action.log(message_type="mapper_created", mapper_type="ensembl_to_symbol", count=len(ensembl_to_symbol))
        mappers['ensembl_to_symbol'] = ensembl_to_symbol
        
        # 2. Symbol -> Ensembl ID (vectorized - much faster than iter_rows)
        action.log(message_type="creating_mapper", mapper_type="symbol_to_ensembl")
        filtered_df = hgnc_df.filter(
            pl.col('symbol').is_not_null() & pl.col('ensembl_gene_id').is_not_null()
        ).select(['symbol', 'ensembl_gene_id'])
        symbol_to_ensembl = dict(zip(
            filtered_df['symbol'].to_list(),
            filtered_df['ensembl_gene_id'].to_list()
        ))
        action.log(message_type="mapper_created", mapper_type="symbol_to_ensembl", count=len(symbol_to_ensembl))
        mappers['symbol_to_ensembl'] = symbol_to_ensembl
        
        # 3. Previous symbols -> Official Symbol
        # This requires splitting strings, so we keep the loop but optimize the filtering
        action.log(message_type="creating_mapper", mapper_type="prev_to_symbol")
        prev_to_symbol = {}
        # Pre-filter to avoid unnecessary iteration
        prev_symbol_df = hgnc_df.filter(
            pl.col('prev_symbol').is_not_null() & pl.col('symbol').is_not_null()
        ).select(['prev_symbol', 'symbol'])
        
        for prev_sym_str, symbol in zip(prev_symbol_df['prev_symbol'].to_list(), 
                                         prev_symbol_df['symbol'].to_list()):
            # Split by comma or pipe
            prev_symbols = str(prev_sym_str).split(',')
            if len(prev_symbols) == 1:
                prev_symbols = str(prev_sym_str).split('|')
            for prev_sym in prev_symbols:
                prev_sym = prev_sym.strip()
                if prev_sym:
                    prev_to_symbol[prev_sym] = symbol
        action.log(message_type="mapper_created", mapper_type="prev_to_symbol", count=len(prev_to_symbol))
        mappers['prev_to_symbol'] = prev_to_symbol
        
        # 4. Alias symbols -> Official Symbol
        # This requires splitting strings, so we keep the loop but optimize the filtering
        action.log(message_type="creating_mapper", mapper_type="alias_to_symbol")
        alias_to_symbol = {}
        # Pre-filter to avoid unnecessary iteration
        alias_symbol_df = hgnc_df.filter(
            pl.col('alias_symbol').is_not_null() & pl.col('symbol').is_not_null()
        ).select(['alias_symbol', 'symbol'])
        
        for alias_sym_str, symbol in zip(alias_symbol_df['alias_symbol'].to_list(),
                                          alias_symbol_df['symbol'].to_list()):
            # Split by comma or pipe
            alias_symbols = str(alias_sym_str).split(',')
            if len(alias_symbols) == 1:
                alias_symbols = str(alias_sym_str).split('|')
            for alias_sym in alias_symbols:
                alias_sym = alias_sym.strip()
                if alias_sym:
                    alias_to_symbol[alias_sym] = symbol
        action.log(message_type="mapper_created", mapper_type="alias_to_symbol", count=len(alias_to_symbol))
        mappers['alias_to_symbol'] = alias_to_symbol
        
        # 5. Combined mapper: any symbol -> official symbol
        action.log(message_type="creating_mapper", mapper_type="any_to_symbol")
        any_to_symbol = {}
        for sym in symbol_to_ensembl.keys():
            any_to_symbol[sym] = sym
        any_to_symbol.update(prev_to_symbol)
        any_to_symbol.update(alias_to_symbol)
        action.log(message_type="mapper_created", mapper_type="any_to_symbol", count=len(any_to_symbol))
        mappers['any_to_symbol'] = any_to_symbol
        
        return mappers


def save_mappers(mappers: Dict[str, Dict[str, str]], output_path: Path) -> None:
    """Save mappers to parquet file.
    
    Stores all mappers in a single parquet file with columns: mapper_type, key, value.
    
    Args:
        mappers: Dictionary containing gene mappers
        output_path: Path to save the parquet file
    """
    with start_action(action_type="save_mappers", output_path=str(output_path)) as action:
        rows = []
        for mapper_type, mapper_dict in mappers.items():
            for key, value in mapper_dict.items():
                rows.append({
                    "mapper_type": mapper_type,
                    "key": key,
                    "value": value
                })
        
        df = pl.DataFrame(rows)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(output_path, compression="zstd", compression_level=3)
        action.log(message_type="mappers_saved", mapper_count=len(mappers), total_entries=len(df))


def load_mappers(input_path: Path) -> Dict[str, Dict[str, str]]:
    """Load mappers from parquet file (or pickle file for backward compatibility).
    
    Args:
        input_path: Path to the parquet or pickle file
        
    Returns:
        Dictionary containing gene mappers
    """
    with start_action(action_type="load_mappers", input_path=str(input_path)) as action:
        # Check if it's a pickle file (backward compatibility)
        if input_path.suffix == '.pkl':
            action.log(message_type="loading_from_pickle", note="backward_compatibility")
            with open(input_path, 'rb') as f:
                mappers = pickle.load(f)
            action.log(message_type="mappers_loaded", mapper_count=len(mappers))
            return mappers
        
        # Load from parquet
        df = pl.read_parquet(input_path)
        
        # Convert back to nested dictionary structure
        mappers: Dict[str, Dict[str, str]] = {}
        for mapper_type in df["mapper_type"].unique().to_list():
            mapper_df = df.filter(pl.col("mapper_type") == mapper_type)
            mappers[mapper_type] = dict(zip(
                mapper_df["key"].to_list(),
                mapper_df["value"].to_list()
            ))
        
        action.log(message_type="mappers_loaded", mapper_count=len(mappers))
        return mappers


def create_hgnc_mapper(shared_dir: Path | None = None) -> Dict[str, Dict[str, str]]:
    """Main function to create HGNC mapper.
    
    Args:
        shared_dir: Directory to save outputs. Defaults to DEFAULT_SHARED_DIR.
        
    Returns:
        Dictionary containing gene mappers
    """
    if shared_dir is None:
        shared_dir = DEFAULT_SHARED_DIR
    
    with start_action(action_type="create_hgnc_mapper", shared_dir=str(shared_dir)) as action:
        shared_dir.mkdir(parents=True, exist_ok=True)
        
        hgnc_df = download_hgnc_data(shared_dir)
        
        if hgnc_df is None:
            action.log(message_type="mapper_creation_failed", reason="download_failed")
            raise RuntimeError("Failed to download HGNC data")
        
        mappers = create_mappers(hgnc_df)
        
        output_path = shared_dir / HGNC_MAPPERS_NAME
        save_mappers(mappers, output_path)
        
        action.log(message_type="mapper_creation_complete", output_path=str(output_path))
        return mappers


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

