"""HGNC gene mapper creation module."""

from pathlib import Path
from typing import Dict
import pickle
import io

import requests
import polars as pl
from eliot import start_action


def download_hgnc_data(output_dir: Path) -> pl.DataFrame | None:
    """Download HGNC complete dataset.
    
    Args:
        output_dir: Directory to save the downloaded data
        
    Returns:
        DataFrame with HGNC data or None if download fails
    """
    with start_action(action_type="download_hgnc_data") as action:
        url = "https://storage.googleapis.com/public-download-files/hgnc/tsv/tsv/hgnc_complete_set.txt"
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; cell2sentence4longevity/0.1)"
        }
        output_file = output_dir / "hgnc_complete_set.tsv"
        
        action.log(message_type="attempt_download", url=url)
        
        try:
            response = requests.get(url, timeout=120, headers=headers)
            response.raise_for_status()
            
            # Use Polars to read the TSV data
            hgnc_df = pl.read_csv(io.StringIO(response.text), separator='\t')
            action.log(message_type="download_success", gene_count=len(hgnc_df))
            
            # Save to file
            hgnc_df.write_csv(output_file, separator='\t')
            action.log(message_type="saved_to_file", path=str(output_file))
            
            return hgnc_df
            
        except Exception as e:
            action.log(message_type="download_failed", error=str(e))
            
            # Check if local file exists as fallback
            action.log(message_type="checking_local_file")
            if output_file.exists():
                hgnc_df = pl.read_csv(output_file, separator='\t')
                action.log(message_type="loaded_from_local", gene_count=len(hgnc_df))
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
        
        # 1. Ensembl ID -> Official Symbol
        action.log(message_type="creating_mapper", mapper_type="ensembl_to_symbol")
        ensembl_to_symbol = {}
        for row in hgnc_df.filter(
            pl.col('ensembl_gene_id').is_not_null() & pl.col('symbol').is_not_null()
        ).iter_rows(named=True):
            ensembl_to_symbol[row['ensembl_gene_id']] = row['symbol']
        action.log(message_type="mapper_created", mapper_type="ensembl_to_symbol", count=len(ensembl_to_symbol))
        mappers['ensembl_to_symbol'] = ensembl_to_symbol
        
        # 2. Symbol -> Ensembl ID
        action.log(message_type="creating_mapper", mapper_type="symbol_to_ensembl")
        symbol_to_ensembl = {}
        for row in hgnc_df.filter(
            pl.col('symbol').is_not_null() & pl.col('ensembl_gene_id').is_not_null()
        ).iter_rows(named=True):
            symbol_to_ensembl[row['symbol']] = row['ensembl_gene_id']
        action.log(message_type="mapper_created", mapper_type="symbol_to_ensembl", count=len(symbol_to_ensembl))
        mappers['symbol_to_ensembl'] = symbol_to_ensembl
        
        # 3. Previous symbols -> Official Symbol
        action.log(message_type="creating_mapper", mapper_type="prev_to_symbol")
        prev_to_symbol = {}
        for row in hgnc_df.filter(
            pl.col('prev_symbol').is_not_null() & pl.col('symbol').is_not_null()
        ).iter_rows(named=True):
            prev_symbols = str(row['prev_symbol']).split(',')
            if len(prev_symbols) == 1:
                prev_symbols = str(row['prev_symbol']).split('|')
            for prev_sym in prev_symbols:
                prev_sym = prev_sym.strip()
                if prev_sym:
                    prev_to_symbol[prev_sym] = row['symbol']
        action.log(message_type="mapper_created", mapper_type="prev_to_symbol", count=len(prev_to_symbol))
        mappers['prev_to_symbol'] = prev_to_symbol
        
        # 4. Alias symbols -> Official Symbol
        action.log(message_type="creating_mapper", mapper_type="alias_to_symbol")
        alias_to_symbol = {}
        for row in hgnc_df.filter(
            pl.col('alias_symbol').is_not_null() & pl.col('symbol').is_not_null()
        ).iter_rows(named=True):
            alias_symbols = str(row['alias_symbol']).split(',')
            if len(alias_symbols) == 1:
                alias_symbols = str(row['alias_symbol']).split('|')
            for alias_sym in alias_symbols:
                alias_sym = alias_sym.strip()
                if alias_sym:
                    alias_to_symbol[alias_sym] = row['symbol']
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
    """Save mappers to pickle file.
    
    Args:
        mappers: Dictionary containing gene mappers
        output_path: Path to save the pickle file
    """
    with start_action(action_type="save_mappers", output_path=str(output_path)) as action:
        with open(output_path, 'wb') as f:
            pickle.dump(mappers, f)
        action.log(message_type="mappers_saved", mapper_count=len(mappers))


def load_mappers(input_path: Path) -> Dict[str, Dict[str, str]]:
    """Load mappers from pickle file.
    
    Args:
        input_path: Path to the pickle file
        
    Returns:
        Dictionary containing gene mappers
    """
    with start_action(action_type="load_mappers", input_path=str(input_path)) as action:
        with open(input_path, 'rb') as f:
            mappers = pickle.load(f)
        action.log(message_type="mappers_loaded", mapper_count=len(mappers))
        return mappers


def create_hgnc_mapper(output_dir: Path) -> Dict[str, Dict[str, str]]:
    """Main function to create HGNC mapper.
    
    Args:
        output_dir: Directory to save outputs
        
    Returns:
        Dictionary containing gene mappers
    """
    with start_action(action_type="create_hgnc_mapper", output_dir=str(output_dir)) as action:
        output_dir.mkdir(parents=True, exist_ok=True)
        
        hgnc_df = download_hgnc_data(output_dir)
        
        if hgnc_df is None:
            action.log(message_type="mapper_creation_failed", reason="download_failed")
            raise RuntimeError("Failed to download HGNC data")
        
        mappers = create_mappers(hgnc_df)
        
        output_path = output_dir / "hgnc_mappers.pkl"
        save_mappers(mappers, output_path)
        
        action.log(message_type="mapper_creation_complete")
        return mappers

