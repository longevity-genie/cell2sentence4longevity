"""CellxGene publication metadata lookup functionality."""

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional

import polars as pl
import requests
from eliot import start_action


# Default cache directory
DEFAULT_CACHE_DIR = Path("./data/shared")
CACHE_FILE_NAME = "cellxgene_collections_cache.parquet"
CACHE_MAX_AGE_DAYS = 7  # Refresh cache if older than 7 days


def _get_cache_path(cache_dir: Path | None = None) -> Path:
    """Get the path to the cache file.
    
    Args:
        cache_dir: Directory to store cache file. Defaults to DEFAULT_CACHE_DIR.
        
    Returns:
        Path to cache file
    """
    if cache_dir is None:
        cache_dir = DEFAULT_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / CACHE_FILE_NAME


def _is_cache_valid(cache_path: Path, max_age_days: int = CACHE_MAX_AGE_DAYS) -> bool:
    """Check if cache file exists and is still valid.
    
    Args:
        cache_path: Path to cache file
        max_age_days: Maximum age of cache in days
        
    Returns:
        True if cache exists and is valid, False otherwise
    """
    if not cache_path.exists():
        return False
    
    cache_age = datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)
    return cache_age < timedelta(days=max_age_days)


def _fetch_all_collections_to_dataframe(
    base_url: str = "https://api.cellxgene.cziscience.com/dp/v1",
    timeout: int = 30,
    max_workers: int = 10
) -> pl.DataFrame:
    """Fetch all collections and their datasets, returning as a DataFrame.
    
    Args:
        base_url: Base API URL
        timeout: Request timeout in seconds
        max_workers: Maximum number of concurrent requests
        
    Returns:
        DataFrame with columns: dataset_id, collection_id, title, doi, description, contact_name, contact_email
    """
    with start_action(action_type="fetch_all_collections") as action:
        # Get all collections
        collections_url = f"{base_url}/collections"
        response = requests.get(collections_url, timeout=timeout)
        response.raise_for_status()
        
        data = response.json()
        collections = data.get("collections", [])
        
        action.log(
            message_type="collections_fetched",
            total_collections=len(collections)
        )
        
        # Fetch details for all collections concurrently
        rows: list[Dict[str, str | None]] = []
        
        def _fetch_collection_details(collection_id: str) -> list[Dict[str, str | None]]:
            """Fetch details for a single collection."""
            detail_url = f"{base_url}/collections/{collection_id}"
            detail_response = requests.get(detail_url, timeout=timeout)
            
            if detail_response.status_code != 200:
                return []
            
            details = detail_response.json()
            datasets = details.get("datasets", [])
            
            collection_rows = []
            for dataset in datasets:
                dataset_id = dataset.get("id") or dataset.get("dataset_id")
                if dataset_id:
                    collection_rows.append({
                        "dataset_id": dataset_id,
                        "collection_id": collection_id,
                        "title": details.get("name", ""),
                        "doi": details.get("doi"),
                        "description": details.get("description", ""),
                        "contact_name": details.get("contact_name", ""),
                        "contact_email": details.get("contact_email", ""),
                    })
            
            return collection_rows
        
        # Use concurrent requests to fetch all collection details
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(_fetch_collection_details, collection["id"])
                for collection in collections
            ]
            
            for future in as_completed(futures):
                collection_rows = future.result()
                rows.extend(collection_rows)
        
        action.log(
            message_type="collections_processed",
            total_datasets=len(rows)
        )
        
        # Create DataFrame
        df = pl.DataFrame(rows)
        return df


def _build_and_save_cache(
    cache_path: Path,
    base_url: str = "https://api.cellxgene.cziscience.com/dp/v1",
    timeout: int = 30,
    max_workers: int = 10
) -> pl.DataFrame:
    """Build cache by fetching all collections and save to parquet.
    
    Args:
        cache_path: Path where cache file should be saved
        base_url: Base API URL
        timeout: Request timeout in seconds
        max_workers: Maximum number of concurrent requests
        
    Returns:
        DataFrame with all collection/dataset mappings
    """
    with start_action(action_type="build_collections_cache") as action:
        df = _fetch_all_collections_to_dataframe(base_url, timeout, max_workers)
        
        # Save to parquet with compression
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(cache_path, compression="zstd", compression_level=3)
        
        action.log(
            message_type="cache_saved",
            cache_path=str(cache_path),
            total_datasets=len(df)
        )
        
        return df


def _load_cache(cache_path: Path) -> pl.DataFrame:
    """Load cache from parquet file.
    
    Args:
        cache_path: Path to cache file
        
    Returns:
        DataFrame with collection/dataset mappings
    """
    return pl.read_parquet(cache_path)


def get_collections_cache(
    cache_dir: Path | None = None,
    force_refresh: bool = False,
    base_url: str = "https://api.cellxgene.cziscience.com/dp/v1",
    timeout: int = 30,
    max_workers: int = 10
) -> pl.DataFrame:
    """Get collections cache, building it if necessary.
    
    Args:
        cache_dir: Directory to store cache file. Defaults to DEFAULT_CACHE_DIR.
        force_refresh: If True, rebuild cache even if valid cache exists
        base_url: Base API URL
        timeout: Request timeout in seconds
        max_workers: Maximum number of concurrent requests
        
    Returns:
        DataFrame with columns: dataset_id, collection_id, title, doi, description, contact_name, contact_email
    """
    cache_path = _get_cache_path(cache_dir)
    
    if force_refresh or not _is_cache_valid(cache_path):
        return _build_and_save_cache(cache_path, base_url, timeout, max_workers)
    
    return _load_cache(cache_path)


def is_cellxgene_dataset(h5ad_path: Path | str) -> bool:
    """Check if a dataset appears to be from CellxGene.
    
    A dataset is considered from CellxGene if:
    1. The filename stem matches a UUID pattern (e.g., 10cc50a0-af80-4fa1-b668-893dd5c0113a)
    2. The path contains 'cellxgene' or 'cziscience' (for URLs)
    
    Args:
        h5ad_path: Path to h5ad file (can be string or Path)
        
    Returns:
        True if dataset appears to be from CellxGene, False otherwise
        
    Examples:
        >>> is_cellxgene_dataset(Path("10cc50a0-af80-4fa1-b668-893dd5c0113a.h5ad"))
        True
        >>> is_cellxgene_dataset(Path("https://datasets.cellxgene.cziscience.com/10cc50a0-af80-4fa1-b668-893dd5c0113a.h5ad"))
        True
        >>> is_cellxgene_dataset(Path("my_dataset.h5ad"))
        False
    """
    path_str = str(h5ad_path).lower()
    
    # Check if path contains cellxgene indicators
    if 'cellxgene' in path_str or 'cziscience' in path_str:
        return True
    
    # Check if filename stem matches UUID pattern
    path_obj = Path(h5ad_path)
    stem = path_obj.stem
    
    # UUID pattern: 8-4-4-4-12 hexadecimal characters
    uuid_pattern = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', re.IGNORECASE)
    if uuid_pattern.match(stem):
        return True
    
    return False


def extract_dataset_id_from_path(h5ad_path: Path) -> str:
    """Extract dataset ID from h5ad file path.
    
    Args:
        h5ad_path: Path to h5ad file
        
    Returns:
        Dataset ID (UUID without .h5ad extension)
        
    Examples:
        >>> extract_dataset_id_from_path(Path("10cc50a0-af80-4fa1-b668-893dd5c0113a.h5ad"))
        '10cc50a0-af80-4fa1-b668-893dd5c0113a'
    """
    return h5ad_path.stem


def dataset_id_exists_in_collections(
    dataset_id: str,
    cache_dir: Path | None = None,
    force_refresh_cache: bool = False,
    timeout: int = 30,
    max_workers: int = 10
) -> bool:
    """Check if a dataset_id exists in the collections cache.
    
    Args:
        dataset_id: Dataset UUID to check
        cache_dir: Directory to store cache file. Defaults to DEFAULT_CACHE_DIR.
        force_refresh_cache: If True, rebuild cache even if valid cache exists
        timeout: Request timeout in seconds (used when building cache)
        max_workers: Maximum number of concurrent requests (used when building cache)
        
    Returns:
        True if dataset_id exists in collections, False otherwise
    """
    with start_action(
        action_type="check_dataset_id_in_collections",
        dataset_id=dataset_id
    ) as action:
        try:
            # Load or build cache
            cache_df = get_collections_cache(
                cache_dir=cache_dir,
                force_refresh=force_refresh_cache,
                timeout=timeout,
                max_workers=max_workers
            )
            
            # Check if dataset_id exists
            exists = len(cache_df.filter(pl.col("dataset_id") == dataset_id)) > 0
            
            action.log(
                message_type="dataset_id_check_complete",
                dataset_id=dataset_id,
                exists_in_collections=exists
            )
            
            return exists
            
        except Exception as e:
            action.log(
                message_type="dataset_id_check_error",
                dataset_id=dataset_id,
                error=str(e),
                error_type=type(e).__name__
            )
            # If we can't check, assume it doesn't exist to be safe
            return False


def join_with_collections(
    df: pl.DataFrame | pl.LazyFrame,
    cache_dir: Path | None = None,
    force_refresh_cache: bool = False,
    timeout: int = 30,
    max_workers: int = 10
) -> pl.DataFrame | pl.LazyFrame:
    """Join a dataframe with collections metadata using dataset_id.
    
    This function performs a left join between the input dataframe and the 
    collections cache, adding publication metadata columns where available.
    
    Args:
        df: Polars DataFrame or LazyFrame with a 'dataset_id' column
        cache_dir: Directory to store cache file. Defaults to DEFAULT_CACHE_DIR.
        force_refresh_cache: If True, rebuild cache even if valid cache exists
        timeout: Request timeout in seconds (used when building cache)
        max_workers: Maximum number of concurrent requests (used when building cache)
        
    Returns:
        DataFrame or LazyFrame with added collection metadata columns:
        - collection_id
        - publication_title (from title)
        - publication_doi (from doi)
        - publication_description (from description)
        - publication_contact_name (from contact_name)
        - publication_contact_email (from contact_email)
        
    Raises:
        ValueError: If 'dataset_id' column is not present in the input dataframe
    """
    with start_action(action_type="join_with_collections") as action:
        # Check if dataset_id column exists
        is_lazy = isinstance(df, pl.LazyFrame)
        columns = df.collect_schema().names() if is_lazy else df.columns
        
        if "dataset_id" not in columns:
            error_msg = "Input dataframe must have a 'dataset_id' column for joining with collections"
            action.log(
                message_type="join_failed",
                error=error_msg,
                available_columns=columns
            )
            raise ValueError(error_msg)
        
        # Load collections cache
        collections_df = get_collections_cache(
            cache_dir=cache_dir,
            force_refresh=force_refresh_cache,
            timeout=timeout,
            max_workers=max_workers
        )
        
        action.log(
            message_type="collections_cache_loaded",
            total_collections=len(collections_df),
            cache_dir=str(cache_dir) if cache_dir else str(DEFAULT_CACHE_DIR)
        )
        
        # Rename columns to match expected output names
        collections_df = collections_df.select([
            pl.col("dataset_id"),
            pl.col("collection_id"),
            pl.col("title").alias("publication_title"),
            pl.col("doi").alias("publication_doi"),
            pl.col("description").alias("publication_description"),
            pl.col("contact_name").alias("publication_contact_name"),
            pl.col("contact_email").alias("publication_contact_email"),
        ])
        
        # Perform left join
        if is_lazy:
            # For LazyFrame, convert collections to lazy and join
            collections_lazy = collections_df.lazy()
            result = df.join(collections_lazy, on="dataset_id", how="left")
        else:
            # For DataFrame, direct join
            result = df.join(collections_df, on="dataset_id", how="left")
        
        # Count matches
        if is_lazy:
            matches = result.select(pl.col("collection_id").is_not_null().sum()).collect().item()
            total = result.select(pl.len()).collect().item()
        else:
            matches = result.filter(pl.col("collection_id").is_not_null()).height
            total = result.height
        
        action.log(
            message_type="join_complete",
            total_rows=total,
            rows_with_metadata=matches,
            match_percentage=round(matches / total * 100, 2) if total > 0 else 0
        )
        
        return result

