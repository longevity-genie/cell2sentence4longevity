"""CLI tool for extracting metadata fields from h5ad AnnData files."""

import gc
import shutil
from pathlib import Path
from typing import Any, Optional

import anndata as ad
import polars as pl
import typer
from eliot import start_action
from pycomfort.logging import to_nice_file, to_nice_stdout
from tqdm import tqdm

from cell2sentence4longevity.preprocessing.h5ad_converter import extract_age_columns
from cell2sentence4longevity.preprocessing.obs_stream import (
    build_obs_chunk_dataframe,
    infer_obs_schema,
    list_obs_columns_from_file,
    list_obs_columns_from_group,
    preload_complex_obs_fields,
)

app = typer.Typer(help="Extract metadata fields from h5ad AnnData files")


MAX_LOG_VALUE_LENGTH = 200
MAX_SAMPLE_ROWS = 5
MAX_NESTED_ITEMS = 8


def _build_summary_expressions(
    schema: pl.Schema,
    total_cells: int
) -> tuple[list[pl.Expr], dict[str, str]]:
    """Build Polars aggregation expressions for creating a summary.
    
    Returns:
        tuple: (list of expressions, dict mapping column names to their source columns for logging)
    """
    exprs: list[pl.Expr] = []
    column_sources: dict[str, str] = {}
    
    # Total cells
    exprs.append(pl.lit(total_cells).alias('total_cells'))
    
    # Dataset ID (first non-null)
    if 'dataset_id' in schema:
        exprs.append(pl.col('dataset_id').filter(pl.col('dataset_id').is_not_null()).first().alias('dataset_id'))
    else:
        exprs.append(pl.lit(None, dtype=pl.String).alias('dataset_id'))
    
    # Organism (first non-null)
    if 'organism' in schema:
        exprs.append(pl.col('organism').filter(pl.col('organism').is_not_null()).first().alias('organism'))
    else:
        exprs.append(pl.lit('Homo sapiens').alias('organism'))
    
    # Unique donors
    donor_cols = ['donor_id', 'donor', 'subject_id', 'individual']
    donor_col = next((col for col in donor_cols if col in schema), None)
    if donor_col:
        exprs.append(pl.col(donor_col).n_unique().alias('unique_donors'))
        column_sources['unique_donors'] = donor_col
    else:
        exprs.append(pl.lit(None, dtype=pl.Int64).alias('unique_donors'))
    
    # Collection info (first non-null values)
    if 'collection_id' in schema:
        exprs.append(pl.col('collection_id').filter(pl.col('collection_id').is_not_null()).first().alias('collection_id'))
    else:
        exprs.append(pl.lit(None, dtype=pl.String).alias('collection_id'))
    
    if 'publication_title' in schema:
        exprs.append(pl.col('publication_title').filter(pl.col('publication_title').is_not_null()).first().alias('publication_title'))
    else:
        exprs.append(pl.lit(None, dtype=pl.String).alias('publication_title'))
    
    if 'publication_doi' in schema:
        exprs.append(pl.col('publication_doi').filter(pl.col('publication_doi').is_not_null()).first().alias('publication_doi'))
    else:
        exprs.append(pl.lit(None, dtype=pl.String).alias('publication_doi'))
    
    # Age statistics
    if 'age_years' in schema:
        exprs.extend([
            pl.col('age_years').min().alias('age_years_min'),
            pl.col('age_years').max().alias('age_years_max'),
            pl.col('age_years').mean().round(2).alias('age_years_mean'),
            pl.col('age_years').is_not_null().sum().alias('cells_with_age_years')
        ])
    else:
        exprs.extend([
            pl.lit(None, dtype=pl.Float64).alias('age_years_min'),
            pl.lit(None, dtype=pl.Float64).alias('age_years_max'),
            pl.lit(None, dtype=pl.Float64).alias('age_years_mean'),
            pl.lit(0).alias('cells_with_age_years')
        ])
    
    if 'age_months' in schema:
        exprs.extend([
            pl.col('age_months').min().alias('age_months_min'),
            pl.col('age_months').max().alias('age_months_max'),
            pl.col('age_months').mean().round(2).alias('age_months_mean'),
            pl.col('age_months').is_not_null().sum().alias('cells_with_age_months')
        ])
    else:
        exprs.extend([
            pl.lit(None, dtype=pl.Float64).alias('age_months_min'),
            pl.lit(None, dtype=pl.Float64).alias('age_months_max'),
            pl.lit(None, dtype=pl.Float64).alias('age_months_mean'),
            pl.lit(0).alias('cells_with_age_months')
        ])
    
    return exprs, column_sources


def _add_categorical_summaries(
    summary_df: pl.DataFrame,
    meta_lazy: pl.LazyFrame,
    schema: pl.Schema
) -> pl.DataFrame:
    """Add categorical summary columns (tissues, cell types, etc.) to summary DataFrame.
    
    This handles the grouped aggregations that can't be done in a single expression.
    """
    # Tissues with counts
    tissue_cols = ['tissue', 'tissue_type', 'organ', 'tissue_general']
    tissue_col = next((col for col in tissue_cols if col in schema), None)
    if tissue_col:
        tissue_counts = (
            meta_lazy
            .group_by(tissue_col)
            .agg(pl.len().alias('count'))
            .sort('count', descending=True)
            .collect(streaming=True)
        )
        unique_tissues = tissue_counts.height
        tissue_list = [
            f"{row[tissue_col]}: {row['count']}"
            for row in tissue_counts.iter_rows(named=True)
            if row[tissue_col] is not None
        ]
        summary_df = summary_df.with_columns([
            pl.lit(unique_tissues).alias('unique_tissues'),
            pl.lit(", ".join(tissue_list) if tissue_list else None).alias('tissues')
        ])
    else:
        summary_df = summary_df.with_columns([
            pl.lit(None, dtype=pl.Int64).alias('unique_tissues'),
            pl.lit(None, dtype=pl.String).alias('tissues')
        ])
    
    # Cell types with counts
    cell_type_cols = ['cell_type', 'cell_type_ontology_term_id', 'celltype']
    cell_type_col = next((col for col in cell_type_cols if col in schema), None)
    if cell_type_col:
        cell_type_counts = (
            meta_lazy
            .group_by(cell_type_col)
            .agg(pl.len().alias('count'))
            .sort('count', descending=True)
            .collect(streaming=True)
        )
        unique_cell_types = cell_type_counts.height
        cell_type_list = [
            f"{row[cell_type_col]}: {row['count']}"
            for row in cell_type_counts.iter_rows(named=True)
            if row[cell_type_col] is not None
        ]
        summary_df = summary_df.with_columns([
            pl.lit(unique_cell_types).alias('unique_cell_types'),
            pl.lit(", ".join(cell_type_list) if cell_type_list else None).alias('cell_types')
        ])
    else:
        summary_df = summary_df.with_columns([
            pl.lit(None, dtype=pl.Int64).alias('unique_cell_types'),
            pl.lit(None, dtype=pl.String).alias('cell_types')
        ])
    
    # Assays with counts
    assay_cols = ['assay', 'assay_ontology_term_id', 'technology', 'platform']
    assay_col = next((col for col in assay_cols if col in schema), None)
    if assay_col:
        assay_counts = (
            meta_lazy
            .group_by(assay_col)
            .agg(pl.len().alias('count'))
            .sort('count', descending=True)
            .collect(streaming=True)
        )
        unique_assays = assay_counts.height
        assay_list = [
            f"{row[assay_col]}: {row['count']}"
            for row in assay_counts.iter_rows(named=True)
            if row[assay_col] is not None
        ]
        summary_df = summary_df.with_columns([
            pl.lit(unique_assays).alias('unique_assays'),
            pl.lit(", ".join(assay_list) if assay_list else None).alias('assays')
        ])
    else:
        summary_df = summary_df.with_columns([
            pl.lit(None, dtype=pl.Int64).alias('unique_assays'),
            pl.lit(None, dtype=pl.String).alias('assays')
        ])
    
    # Sex with counts
    sex_cols = ['sex', 'sex_ontology_term_id', 'gender']
    sex_col = next((col for col in sex_cols if col in schema), None)
    if sex_col:
        sex_counts = (
            meta_lazy
            .group_by(sex_col)
            .agg(pl.len().alias('count'))
            .sort('count', descending=True)
            .collect(streaming=True)
        )
        unique_sexes = sex_counts.height
        sex_list = [
            f"{row[sex_col]}: {row['count']}"
            for row in sex_counts.iter_rows(named=True)
            if row[sex_col] is not None
        ]
        summary_df = summary_df.with_columns([
            pl.lit(unique_sexes).alias('unique_sexes'),
            pl.lit(", ".join(sex_list) if sex_list else None).alias('sexes')
        ])
    else:
        summary_df = summary_df.with_columns([
            pl.lit(None, dtype=pl.Int64).alias('unique_sexes'),
            pl.lit(None, dtype=pl.String).alias('sexes')
        ])
    
    # Diseases with counts
    disease_cols = ['disease', 'disease_ontology_term_id', 'condition']
    disease_col = next((col for col in disease_cols if col in schema), None)
    if disease_col:
        disease_counts = (
            meta_lazy
            .group_by(disease_col)
            .agg(pl.len().alias('count'))
            .sort('count', descending=True)
            .collect(streaming=True)
        )
        unique_diseases = disease_counts.height
        disease_list = [
            f"{row[disease_col]}: {row['count']}"
            for row in disease_counts.iter_rows(named=True)
            if row[disease_col] is not None
        ]
        summary_df = summary_df.with_columns([
            pl.lit(unique_diseases).alias('unique_diseases'),
            pl.lit(", ".join(disease_list) if disease_list else None).alias('diseases')
        ])
    else:
        summary_df = summary_df.with_columns([
            pl.lit(None, dtype=pl.Int64).alias('unique_diseases'),
            pl.lit(None, dtype=pl.String).alias('diseases')
        ])
    
    # Add unique ages as comma-separated strings with counts
    if 'age_years' in schema:
        age_years_counts = (
            meta_lazy
            .filter(pl.col('age_years').is_not_null())
            .group_by('age_years')
            .agg(pl.len().alias('count'))
            .sort('age_years')
            .collect(streaming=True)
        )
        age_years_list = [
            f"{row['age_years']}: {row['count']}"
            for row in age_years_counts.iter_rows(named=True)
        ]
        summary_df = summary_df.with_columns(
            pl.lit(", ".join(age_years_list) if age_years_list else None).alias('unique_ages_years')
        )
    else:
        summary_df = summary_df.with_columns(
            pl.lit(None, dtype=pl.String).alias('unique_ages_years')
        )
    
    if 'age_months' in schema:
        age_months_counts = (
            meta_lazy
            .filter(pl.col('age_months').is_not_null())
            .group_by('age_months')
            .agg(pl.len().alias('count'))
            .sort('age_months')
            .collect(streaming=True)
        )
        age_months_list = [
            f"{row['age_months']}: {row['count']}"
            for row in age_months_counts.iter_rows(named=True)
        ]
        summary_df = summary_df.with_columns(
            pl.lit(", ".join(age_months_list) if age_months_list else None).alias('unique_ages_months')
        )
    else:
        summary_df = summary_df.with_columns(
            pl.lit(None, dtype=pl.String).alias('unique_ages_months')
        )
    
    return summary_df


def _mask_age_months_for_non_mouse(summary_df: pl.DataFrame) -> pl.DataFrame:
    """Ensure age-in-months summary columns are only populated for mouse datasets.
    
    For rows where organism is not 'Mus musculus', all age_months-related columns
    are set to null while keeping the columns present in the schema.
    """
    if 'organism' not in summary_df.columns:
        return summary_df

    age_months_columns = [
        'age_months_min',
        'age_months_max',
        'age_months_mean',
        'cells_with_age_months',
        'unique_ages_months',
    ]
    existing_age_months_columns = [
        column for column in age_months_columns if column in summary_df.columns
    ]
    if not existing_age_months_columns:
        return summary_df

    return summary_df.with_columns([
        pl.when(pl.col('organism') == 'Mus musculus')
        .then(pl.col(column))
        .otherwise(None)
        .alias(column)
        for column in existing_age_months_columns
    ])


def _coerce_all_null_object_columns(
    df: pl.DataFrame,
    obs_schema: dict[str, pl.DataType]
) -> pl.DataFrame:
    if not df.columns:
        return df
    replacements: list[pl.Expr] = []
    for column_name, dtype in df.schema.items():
        if dtype != pl.Object:
            continue
        if column_name not in obs_schema:
            continue
        series = df[column_name]
        if series.is_null().all():
            target_dtype = obs_schema[column_name]
            replacements.append(pl.lit(None, dtype=target_dtype).alias(column_name))
    if not replacements:
        return df
    return df.with_columns(replacements)


def _truncate_for_log(value: str) -> str:
    if len(value) <= MAX_LOG_VALUE_LENGTH:
        return value
    return f"{value[:MAX_LOG_VALUE_LENGTH]}...[truncated]"


def _safe_serialize_for_log(value: Any, *, max_items: int = MAX_NESTED_ITEMS) -> Any:
    if value is None:
        return None
    if isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return _truncate_for_log(value)
    if isinstance(value, (bytes, bytearray)):
        decoded = value.decode("utf-8", errors="replace")
        return _truncate_for_log(decoded)
    if isinstance(value, dict):
        limited_items = list(value.items())[:max_items]
        return {
            str(k): _safe_serialize_for_log(v, max_items=max_items)
            for k, v in limited_items
        }
    if isinstance(value, (list, tuple, set)):
        limited_values = list(value)[:max_items]
        return [_safe_serialize_for_log(v, max_items=max_items) for v in limited_values]
    if hasattr(value, "tolist"):
        try:
            data_list = value.tolist()
            return _safe_serialize_for_log(data_list, max_items=max_items)
        except Exception:
            pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    representation = repr(value)
    return _truncate_for_log(representation)


def _collect_sample_rows(df: pl.DataFrame, max_rows: int = MAX_SAMPLE_ROWS) -> list[dict[str, Any]]:
    if df.height == 0:
        return []
    sample_rows: list[dict[str, Any]] = []
    for row in df.head(max_rows).iter_rows(named=True):
        serialized_row = {column: _safe_serialize_for_log(value) for column, value in row.items()}
        sample_rows.append(serialized_row)
    return sample_rows


def extract_fields_from_h5ad(
    h5ad_path: Path,
    fields: Optional[list[str]],
    output_path: Path,
    chunk_size: int = 10000,
    compression: str = "zstd",
    compression_level: int = 3,
    use_pyarrow: bool = True,
    extract_age: bool = True,
    age_source_col: str = "development_stage",
    generate_summary: bool = False,
    verbose_logging: bool = True
) -> None:
    """Extract specified metadata fields from h5ad file in a memory-efficient way.
    
    Args:
        h5ad_path: Path to h5ad file
        fields: List of field names to extract from adata.obs (None = all fields)
        output_path: Path where to save the parquet file
        chunk_size: Number of rows to process at a time
        compression: Compression algorithm (zstd, snappy, gzip, etc.)
        compression_level: Compression level (1-22 for zstd)
        use_pyarrow: Whether to use PyArrow for parquet writing
        extract_age: Whether to extract age from development_stage column
        age_source_col: Column to extract age from (default: development_stage)
        generate_summary: Whether to generate a summary with unique values and counts
        verbose_logging: Whether to log detailed chunk processing info (disable for batch processing)
    """
    with start_action(
        action_type="extract_fields_from_h5ad",
        h5ad_path=str(h5ad_path),
        fields=fields if fields is not None else "all_fields",
        output_path=str(output_path)
    ) as action:
        # Load h5ad in backed mode (memory efficient)
        adata = ad.read_h5ad(h5ad_path, backed='r')
        n_cells = adata.n_obs
        obs_group = adata.file["obs"]
        obs_columns = list_obs_columns_from_group(obs_group)
        obs_schema = infer_obs_schema(obs_group)
        string_fields = {col for col, dtype in obs_schema.items() if dtype == pl.String}
        requested_fields = fields if fields is not None else obs_columns
        preloaded_fields = preload_complex_obs_fields(
            obs_group=obs_group,
            fields=[field for field in requested_fields if field in obs_columns],
            total_rows=n_cells
        )
        if preloaded_fields and verbose_logging:
            action.log(
                message_type="preloaded_complex_obs_fields",
                fields=list(preloaded_fields.keys())
            )
        
        if verbose_logging:
            action.log(
                message_type="h5ad_loaded",
                n_cells=n_cells,
                n_genes=adata.n_vars,
                available_fields=obs_columns
            )
        
        # Check which fields are available
        available_fields = set(obs_columns)
        missing_fields = [f for f in requested_fields if f not in available_fields]
        valid_fields = [f for f in requested_fields if f in available_fields]
        
        if missing_fields and verbose_logging:
            action.log(
                message_type="missing_fields_warning",
                missing_fields=missing_fields,
                valid_fields=valid_fields
            )
        
        if not valid_fields:
            action.log(
                message_type="error_no_valid_fields",
                error="None of the requested fields are available in the h5ad file"
            )
            raise ValueError(f"None of the requested fields {fields} are available. Available fields: {list(available_fields)}")
        
        # Process in chunks for memory efficiency
        n_chunks = (n_cells + chunk_size - 1) // chunk_size
        if verbose_logging:
            action.log(
                message_type="starting_chunk_processing",
                total_chunks=n_chunks,
                chunk_size=chunk_size
            )
        
        # Create temp directory for chunks
        temp_dir = output_path.parent / f".temp_{output_path.stem}"
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        categorical_cache: dict[str, Any] = {}
        
        # Only show progress bar if verbose logging is enabled
        chunk_range = tqdm(range(n_chunks), desc='Extracting metadata', disable=not verbose_logging)
        for chunk_idx in chunk_range:
            # Only create chunk action if verbose logging is enabled
            if verbose_logging:
                chunk_context = start_action(action_type="process_chunk", chunk_idx=chunk_idx)
            else:
                # Use a dummy context manager that does nothing
                from contextlib import nullcontext
                chunk_context = nullcontext()
            
            with chunk_context as chunk_action:
                start_idx = chunk_idx * chunk_size
                end_idx = min(start_idx + chunk_size, n_cells)
                
                chunk_df: pl.DataFrame | None = None
                chunk_columns: list[str] = []
                
                try:
                    chunk_df = build_obs_chunk_dataframe(
                        obs_group=obs_group,
                        fields=valid_fields,
                        start_idx=start_idx,
                        end_idx=end_idx,
                        categorical_cache=categorical_cache,
                        string_fields=string_fields,
                    string_fill_value=None,
                    preloaded_fields=preloaded_fields
                    )
                    chunk_df = _coerce_all_null_object_columns(chunk_df, obs_schema)
                    
                    chunk_columns = list(chunk_df.columns)
                    
                    cast_exprs = [
                        pl.col(col).cast(dtype, strict=False)
                        for col, dtype in obs_schema.items()
                        if col in chunk_columns
                    ]
                    if cast_exprs:
                        chunk_df = chunk_df.with_columns(cast_exprs)
                    fill_exprs = [
                        pl.col(col).fill_null("nan")
                        for col, dtype in obs_schema.items()
                        if dtype == pl.String and col in chunk_columns
                    ]
                    if fill_exprs:
                        chunk_df = chunk_df.with_columns(fill_exprs)
                    
                    if extract_age:
                        chunk_df = extract_age_columns(
                            chunk_df,
                            development_stage_col=age_source_col
                        )
                        # Ensure both age columns exist in all chunks for consistent schema
                        # This prevents schema mismatch errors when scanning parquet files
                        age_columns_to_add = []
                        if 'age_years' not in chunk_df.columns:
                            age_columns_to_add.append(pl.lit(None).cast(pl.Float64).alias('age_years'))
                        if 'age_months' not in chunk_df.columns:
                            age_columns_to_add.append(pl.lit(None).cast(pl.Float64).alias('age_months'))
                        if age_columns_to_add:
                            chunk_df = chunk_df.with_columns(age_columns_to_add)
                    
                    # Write chunk to temp file
                    chunk_file = temp_dir / f"chunk_{chunk_idx:04d}.parquet"
                    chunk_df.write_parquet(
                        chunk_file,
                        compression=compression,
                        compression_level=compression_level,
                        use_pyarrow=use_pyarrow
                    )
                    
                    if verbose_logging and chunk_action is not None:
                        chunk_action.log(
                            message_type="chunk_processed",
                            rows=end_idx - start_idx,
                            start_idx=start_idx,
                            end_idx=end_idx
                        )
                    
                    chunk_df = None
                except Exception as exc:
                    sample_rows: list[dict[str, Any]] = []
                    if chunk_df is not None:
                        try:
                            sample_rows = _collect_sample_rows(chunk_df)
                        except Exception as sample_exc:
                            sample_rows = [
                                {
                                    "_error": f"Failed to serialize sample rows: {sample_exc}"
                                }
                            ]
                    if verbose_logging and chunk_action is not None:
                        chunk_action.log(
                            message_type="chunk_failure_sample",
                            error=str(exc),
                            start_idx=start_idx,
                            end_idx=end_idx,
                            columns=chunk_columns,
                            sample_rows=sample_rows
                        )
                    raise
                finally:
                    if chunk_df is not None:
                        del chunk_df
        
        # Read all chunks lazily without materializing full dataset
        if verbose_logging:
            action.log(message_type="building_lazy_dataframe", n_chunks=n_chunks)
        try:
            final_lazy_df = pl.scan_parquet(temp_dir / "*.parquet")
            schema_names = set(final_lazy_df.collect_schema().names())
        except Exception as e:
            error_msg = str(e)
            # Extract column name from Polars schema mismatch errors
            column_name = None
            if "extra column in file outside of expected schema:" in error_msg:
                # Format: "extra column in file outside of expected schema: column_name, hint: ..."
                parts = error_msg.split("extra column in file outside of expected schema:")
                if len(parts) > 1:
                    column_part = parts[1].split(",")[0].strip()
                    column_name = column_part
            elif "missing column" in error_msg.lower():
                # Try to extract column name from missing column errors
                parts = error_msg.split("missing column")
                if len(parts) > 1:
                    column_part = parts[1].split()[0].strip().rstrip(":")
                    column_name = column_part
            
            if column_name:
                action.log(
                    message_type="schema_mismatch_error",
                    error=error_msg,
                    column_name=column_name,
                    temp_dir=str(temp_dir)
                )
                raise ValueError(
                    f"Schema mismatch when reading parquet chunks: column '{column_name}' is present in some chunks but not others. "
                    f"Original error: {error_msg}"
                ) from e
            else:
                action.log(
                    message_type="parquet_scan_error",
                    error=error_msg,
                    temp_dir=str(temp_dir)
                )
                raise
        
        # Join with collections if applicable
        collection_columns = ['collection_id', 'publication_title', 'publication_doi']
        has_collection_data = any(col in schema_names for col in collection_columns)
        dataset_id: Optional[str] = None
        
        if not has_collection_data:
            from cell2sentence4longevity.preprocessing.publication_lookup import (
                is_cellxgene_dataset,
                extract_dataset_id_from_path,
                dataset_id_exists_in_collections,
                join_with_collections
            )
            
            # Check if this is a CellxGene dataset
            if is_cellxgene_dataset(h5ad_path):
                dataset_id = extract_dataset_id_from_path(h5ad_path)
                
                # Add dataset_id column if not present
                if 'dataset_id' not in schema_names:
                    final_lazy_df = final_lazy_df.with_columns(pl.lit(dataset_id).alias('dataset_id'))
                    schema_names = set(final_lazy_df.collect_schema().names())
                    if verbose_logging:
                        action.log(
                            message_type="dataset_id_added",
                            dataset_id=dataset_id
                        )
                
                # Always attempt to join with collections (left join will add null values if not found)
                # This ensures consistent schema across all summaries
                if verbose_logging:
                    action.log(
                        message_type="joining_with_collections",
                        dataset_id=dataset_id
                    )
                final_lazy_df = join_with_collections(final_lazy_df)
                schema_names = set(final_lazy_df.collect_schema().names())
                
                # Check if dataset was actually found in collections
                if dataset_id_exists_in_collections(dataset_id):
                    if verbose_logging:
                        action.log(
                            message_type="collection_found",
                            dataset_id=dataset_id,
                            note="Dataset found in collections, metadata added"
                        )
                else:
                    if verbose_logging:
                        action.log(
                            message_type="dataset_not_in_collections",
                            dataset_id=dataset_id,
                            note="Dataset not found in collections cache, collection columns will be null"
                        )
            else:
                if verbose_logging:
                    action.log(
                        message_type="not_cellxgene_dataset",
                        note="Dataset does not match CellxGene UUID pattern, skipping collection join"
                    )
        
        final_columns = list(schema_names)
        
        # Write to parquet using sink_parquet for memory efficiency
        action.log(
            message_type="writing_parquet",
            output_path=str(output_path),
            rows=n_cells,
            columns=final_columns
        )
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Use lazy API with sink_parquet for memory-efficient writing
        final_lazy_df.sink_parquet(
            output_path,
            compression=compression,
            compression_level=compression_level
        )
        
        # Clean up temp directory
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
            if verbose_logging:
                action.log(message_type="temp_dir_cleaned", temp_dir=str(temp_dir))
        
        written_lazy_df = pl.scan_parquet(output_path)
        written_schema = written_lazy_df.collect_schema()
        
        if extract_age:
            age_exprs = []
            if 'age_years' in written_schema:
                age_exprs.append(pl.col('age_years').is_not_null().sum().alias('non_null_age_years'))
            if 'age_months' in written_schema:
                age_exprs.append(pl.col('age_months').is_not_null().sum().alias('non_null_age_months'))
            
            if age_exprs:
                age_counts = written_lazy_df.select(age_exprs).collect(streaming=True)
                log_data: dict[str, Any] = {"message_type": "age_extracted"}
                if 'non_null_age_years' in age_counts.columns:
                    log_data["non_null_age_years"] = int(age_counts['non_null_age_years'][0])
                if 'non_null_age_months' in age_counts.columns:
                    log_data["non_null_age_months"] = int(age_counts['non_null_age_months'][0])
                action.log(**log_data)
        
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        action.log(
            message_type="extraction_complete",
            output_path=str(output_path),
            rows=n_cells,
            columns=len(final_columns),
            size_mb=round(file_size_mb, 2)
        )
        
        # Generate summary if requested
        if generate_summary:
            summary_path = output_path.with_name(output_path.stem + "_summary.csv")
            summary_lazy = written_lazy_df
            summary_schema = summary_lazy.collect_schema()
            
            with start_action(action_type="generate_summary", summary_path=str(summary_path)) as summary_action:
                # Build aggregation expressions for simple stats
                exprs, column_sources = _build_summary_expressions(summary_schema, n_cells)
                
                # Create base summary with simple aggregations
                summary_df = summary_lazy.select(exprs).collect(streaming=True)
                
                # Add categorical summaries (tissues, cell types, etc.)
                summary_df = _add_categorical_summaries(summary_df, summary_lazy, summary_schema)
                summary_df = _mask_age_months_for_non_mouse(summary_df)
                
                # Log statistics
                if 'dataset_id' in summary_df.columns:
                    dataset_id_val = summary_df['dataset_id'][0]
                    if dataset_id_val is not None:
                        summary_action.log(message_type="dataset_id_in_summary", dataset_id=dataset_id_val)
                
                for col_name, source_col in column_sources.items():
                    if col_name in summary_df.columns:
                        val = summary_df[col_name][0]
                        if val is not None:
                            summary_action.log(message_type=f"{col_name}_info", column=source_col, value=val)
                
                # Write summary
                summary_df.write_csv(summary_path)
                
                summary_action.log(
                    message_type="summary_complete",
                    path=str(summary_path),
                    columns_count=len(summary_df.columns)
                )
        
        # Close the h5ad file and clean up
        adata.file.close()
        del adata, final_lazy_df, written_lazy_df
        if verbose_logging:
            action.log(message_type="h5ad_file_closed")


@app.command()
def extract(
    h5ad_path: Path = typer.Argument(..., help="Path to h5ad file"),
    fields: Optional[list[str]] = typer.Option(None, "--field", "-f", help="Field name to extract (can be specified multiple times). If not specified, extracts all available metadata fields."),
    output_path: Optional[Path] = typer.Option(None, "--output", "-o", help="Output parquet file path (default: data/output/meta/<h5ad_name>_meta.parquet)"),
    chunk_size: int = typer.Option(10000, "--chunk-size", help="Number of rows to process at a time"),
    compression: str = typer.Option("zstd", "--compression", help="Compression algorithm"),
    compression_level: int = typer.Option(3, "--compression-level", help="Compression level"),
    extract_age: bool = typer.Option(True, "--extract-age/--no-extract-age", help="Extract age from development_stage column"),
    age_source_col: str = typer.Option("development_stage", "--age-source", help="Column to extract age from"),
    generate_summary: bool = typer.Option(False, "--summary/--no-summary", help="Generate summary with unique tissues, cell types, ages, and counts"),
    log_dir: Optional[Path] = typer.Option(None, "--log-dir", help="Directory for log files"),
) -> None:
    """Extract metadata fields from a single h5ad file.
    
    By default, extracts all available metadata fields from adata.obs.
    Specify -f/--field to extract only specific fields.
    
    Examples:
        # Extract all metadata fields to default location (data/output/meta/)
        explore extract data.h5ad
        
        # Extract specific fields only
        explore extract data.h5ad -f age -f cell_type -f tissue
        
        # Specify custom output path
        explore extract data.h5ad -o custom/path/output.parquet
    """
    # Setup logging
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
        json_log = log_dir / "extract.json"
        rendered_log = log_dir / "extract.log"
        to_nice_file(output_file=json_log, rendered_file=rendered_log)
        to_nice_stdout(output_file=json_log)
    else:
        # Default to ./logs directory
        default_log_dir = Path("./logs")
        default_log_dir.mkdir(parents=True, exist_ok=True)
        json_log = default_log_dir / "extract.json"
        rendered_log = default_log_dir / "extract.log"
        to_nice_file(output_file=json_log, rendered_file=rendered_log)
        to_nice_stdout(output_file=json_log)
    
    # Determine output path
    if output_path is None:
        output_dir = Path("data/output/meta")
        output_path = output_dir / f"{h5ad_path.stem}_meta.parquet"
        
        # Create directory if it doesn't exist
        if not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)
            typer.echo(f"Created output directory: {output_dir}")
    
    # Get fields to extract
    if fields is None:
        fields_to_extract = list_obs_columns_from_file(h5ad_path)
        typer.echo(f"Extracting all {len(fields_to_extract)} available metadata fields")
    else:
        fields_to_extract = fields
        typer.echo(f"Extracting {len(fields_to_extract)} specified fields")
    
    # Extract fields
    extract_fields_from_h5ad(
        h5ad_path=h5ad_path,
        fields=fields_to_extract,
        output_path=output_path,
        chunk_size=chunk_size,
        compression=compression,
        compression_level=compression_level,
        use_pyarrow=True,
        extract_age=extract_age,
        age_source_col=age_source_col,
        generate_summary=generate_summary
    )
    
    age_info = " (with age extraction)" if extract_age else ""
    summary_info = " and summary" if generate_summary else ""
    typer.echo(f"✓ Extracted {len(fields_to_extract)} fields to {output_path}{age_info}{summary_info}")


@app.command()
def batch(
    input_dir: Optional[Path] = typer.Argument(None, help="Directory containing h5ad files (default: ./data/input)"),
    output_dir: Optional[Path] = typer.Option(None, "--output-dir", "-o", help="Output directory for parquet files (default: data/output/meta)"),
    fields: Optional[list[str]] = typer.Option(None, "--field", "-f", help="Field name to extract (can be specified multiple times). If not specified, extracts all available metadata fields."),
    chunk_size: int = typer.Option(10000, "--chunk-size", help="Number of rows to process at a time"),
    compression: str = typer.Option("zstd", "--compression", help="Compression algorithm"),
    compression_level: int = typer.Option(3, "--compression-level", help="Compression level"),
    extract_age: bool = typer.Option(True, "--extract-age/--no-extract-age", help="Extract age from development_stage column"),
    age_source_col: str = typer.Option("development_stage", "--age-source", help="Column to extract age from"),
    generate_summary: bool = typer.Option(True, "--summary/--no-summary", help="Generate species-level summary files"),
    summary_format: str = typer.Option("csv", "--summary-format", help="Format for summary files: 'csv' or 'tsv' (default: csv)"),
    log_dir: Optional[Path] = typer.Option(None, "--log-dir", help="Directory for log files"),
    skip_existing: bool = typer.Option(True, "--skip-existing/--overwrite", help="Skip files that already have output"),
    max_threads: Optional[int] = typer.Option(None, "--max-threads", help="Maximum number of threads for Polars (default: 8, set to 1 for minimal memory)"),
    max_file_size_mb: Optional[float] = typer.Option(None, "--max-file-size-mb", help="Maximum file size in MB to process (e.g., 12000 for 12 GB). Files larger than this will be skipped."),
    verbose_logging: bool = typer.Option(False, "--verbose/--quiet", help="Enable verbose logging including per-chunk details (default: quiet for batch)"),
) -> None:
    """Extract metadata fields from multiple h5ad files in batch.
    
    By default, extracts all available metadata fields from adata.obs from ./data/input directory.
    Specify -f/--field to extract only specific fields.
    
    Logging: By default, batch mode uses quiet logging (only per-file summaries and errors).
    Use --verbose to enable detailed per-chunk logging.
    
    Examples:
        # Extract all metadata fields from default input directory (./data/input)
        explore batch
        
        # Extract from specific directory
        explore batch ./my/custom/input
        
        # Extract specific fields only
        explore batch -f age -f cell_type -f tissue
        
        # Specify custom input and output directories
        explore batch ./data/input --output-dir ./custom/output
        
        # Enable verbose logging with per-chunk details
        explore batch --verbose
    """
    # Limit Polars thread pool to prevent memory overload with many files
    # Use half of available cores by default to balance speed and memory
    import os
    if max_threads is not None:
        threads = max_threads
    else:
        cpu_count = os.cpu_count() or 4
        threads = max(1, cpu_count // 2)  # Half of cores, minimum 1
    os.environ['POLARS_MAX_THREADS'] = str(threads)
    typer.echo(f"Using {threads} threads for Polars operations")
    
    # Use default input directory if not specified
    if input_dir is None:
        input_dir = Path("./data/input")
    
    # Determine output directory
    if output_dir is None:
        output_dir = Path("data/output/meta")
    
    # Setup logging with datetime stamp to preserve logs from multiple runs
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
        json_log = log_dir / f"batch_extract_{timestamp}.json"
        rendered_log = log_dir / f"batch_extract_{timestamp}.log"
        to_nice_file(output_file=json_log, rendered_file=rendered_log)
        to_nice_stdout(output_file=json_log)
    else:
        # Default to ./logs directory
        default_log_dir = Path("./logs")
        default_log_dir.mkdir(parents=True, exist_ok=True)
        json_log = default_log_dir / f"batch_extract_{timestamp}.json"
        rendered_log = default_log_dir / f"batch_extract_{timestamp}.log"
        to_nice_file(output_file=json_log, rendered_file=rendered_log)
        to_nice_stdout(output_file=json_log)
    
    with start_action(
        action_type="batch_extract_fields",
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        fields=fields if fields else "all_fields",
        verbose_logging=verbose_logging
    ) as action:
        # Find all h5ad files
        h5ad_files = list(input_dir.glob("*.h5ad"))
        
        if not h5ad_files:
            action.log(
                message_type="error_no_files",
                error=f"No h5ad files found in {input_dir}"
            )
            typer.echo(f"✗ No h5ad files found in {input_dir}", err=True)
            raise typer.Exit(code=1)
        
        action.log(
            message_type="found_h5ad_files",
            count=len(h5ad_files),
            files=[f.name for f in h5ad_files]
        )
        
        # Create output directory
        if not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)
            action.log(
                message_type="created_output_directory",
                directory=str(output_dir)
            )
            typer.echo(f"Created output directory: {output_dir}")
        else:
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # Process each file
        results: list[tuple[str, bool, str]] = []
        
        for idx, h5ad_file in enumerate(h5ad_files, 1):
            dataset_name = h5ad_file.stem
            output_path = output_dir / f"{dataset_name}_meta.parquet"
            
            # Check file size if limit is specified
            if max_file_size_mb is not None:
                file_size_mb = h5ad_file.stat().st_size / (1024 * 1024)
                if file_size_mb > max_file_size_mb:
                    action.log(
                        message_type="skipping_large_file",
                        file=h5ad_file.name,
                        file_size_mb=round(file_size_mb, 2),
                        max_file_size_mb=max_file_size_mb
                    )
                    typer.echo(f"[{idx}/{len(h5ad_files)}] Skipping {h5ad_file.name} (file size {file_size_mb:.2f} MB exceeds limit {max_file_size_mb} MB)")
                    results.append((dataset_name, False, f"skipped (file too large: {file_size_mb:.2f} MB > {max_file_size_mb} MB)"))
                    continue
            
            # Skip if output exists and skip_existing is True
            if skip_existing and output_path.exists():
                action.log(
                    message_type="skipping_existing",
                    file=h5ad_file.name,
                    output_path=str(output_path)
                )
                typer.echo(f"[{idx}/{len(h5ad_files)}] Skipping {h5ad_file.name} (output exists)")
                results.append((dataset_name, True, "skipped (exists)"))
                continue
            
            typer.echo(f"\n[{idx}/{len(h5ad_files)}] Processing {h5ad_file.name}...")
            
            with start_action(
                action_type="process_file",
                file=h5ad_file.name,
                index=idx,
                total=len(h5ad_files)
            ) as file_action:
                success = False
                message = ""
                
                try:
                    # Get fields for this file
                    if fields is None:
                        fields_to_extract = list_obs_columns_from_file(h5ad_file)
                        file_action.log(
                            message_type="extracting_all_fields",
                            n_fields=len(fields_to_extract)
                        )
                    else:
                        fields_to_extract = fields
                    
                    extract_fields_from_h5ad(
                        h5ad_path=h5ad_file,
                        fields=fields_to_extract,
                        output_path=output_path,
                        chunk_size=chunk_size,
                        compression=compression,
                        compression_level=compression_level,
                        use_pyarrow=True,
                        extract_age=extract_age,
                        age_source_col=age_source_col,
                        generate_summary=False,  # Never generate individual summaries in batch mode
                        verbose_logging=verbose_logging  # Use user-specified verbosity (default: False for batch)
                    )
                    success = True
                    message = "success"
                    age_info = " (with age)" if extract_age else ""
                    typer.echo(f"  ✓ Extracted to {output_path.name}{age_info}")
                    
                except Exception as e:
                    success = False
                    message = str(e)
                    file_action.log(
                        message_type="error_processing_file",
                        error=message,
                        file=h5ad_file.name
                    )
                    typer.echo(f"  ✗ Error: {message}", err=True)
                
                results.append((dataset_name, success, message))
                
                # Force garbage collection after each file
                gc.collect()
        
        # Summary
        successful = sum(1 for _, success, _ in results if success)
        failed = len(results) - successful
        
        action.log(
            message_type="batch_complete",
            total_files=len(h5ad_files),
            successful=successful,
            failed=failed
        )
        
        # Generate species-level summary if requested
        if generate_summary and successful > 0:
            # Validate format
            summary_format_lower = summary_format.lower()
            if summary_format_lower not in ['csv', 'tsv']:
                typer.echo(f"✗ Invalid summary format: {summary_format}. Must be 'csv' or 'tsv'", err=True)
                raise typer.Exit(code=1)
            
            # Set separator based on format
            separator = ',' if summary_format_lower == 'csv' else '\t'
            
            with start_action(action_type="generate_species_summaries", output_dir=str(output_dir), format=summary_format_lower) as summary_action:
                typer.echo(f"\nGenerating species-level summaries ({summary_format_lower.upper()})...")
                
                # Find all metadata files
                meta_files = list(output_dir.glob("*_meta.parquet"))
                
                if meta_files:
                    summary_action.log(
                        message_type="found_meta_files",
                        count=len(meta_files)
                    )
                    
                    # Generate summaries for each metadata file
                    summary_temp_dir = output_dir / ".temp_summaries"
                    summary_temp_dir.mkdir(parents=True, exist_ok=True)
                    
                    for idx, meta_file in enumerate(meta_files):
                        dataset_id = meta_file.stem.replace("_meta", "")
                        
                        try:
                            # Use lazy API - don't collect until necessary
                            meta_lazy = pl.scan_parquet(meta_file)
                            meta_schema = meta_lazy.collect_schema()
                            
                            # Get total cell count
                            total_cells = meta_lazy.select(pl.len()).collect(streaming=True).item()
                            
                            # Build aggregation expressions for simple stats
                            exprs, _ = _build_summary_expressions(meta_schema, total_cells)
                            
                            # Override dataset_id with the one from filename (first expression is dataset_id)
                            # Replace the dataset_id expression with literal from filename
                            exprs_updated = []
                            for expr in exprs:
                                # Check if this is the dataset_id column by trying to get its alias
                                try:
                                    output_name = expr.meta.output_name()
                                    if output_name == 'dataset_id':
                                        exprs_updated.append(pl.lit(dataset_id).alias('dataset_id'))
                                    else:
                                        exprs_updated.append(expr)
                                except Exception:
                                    exprs_updated.append(expr)
                            
                            # Create base summary with simple aggregations
                            summary_df = meta_lazy.select(exprs_updated).collect(streaming=True)
                            
                            # Add categorical summaries (tissues, cell types, etc.)
                            summary_df = _add_categorical_summaries(summary_df, meta_lazy, meta_schema)
                            summary_df = _mask_age_months_for_non_mouse(summary_df)
                            
                            # Write summary to temp file
                            summary_file = summary_temp_dir / f"summary_{idx:04d}.parquet"
                            summary_df.write_parquet(summary_file)
                            
                            # Clean up
                            del meta_lazy, summary_df
                            
                        except Exception as e:
                            summary_action.log(
                                message_type="error_creating_summary",
                                file=meta_file.name,
                                error=str(e)
                            )
                    
                    # Read all summaries using lazy API but handle schema differences
                    summary_files = sorted(summary_temp_dir.glob("summary_*.parquet"))
                    if summary_files:
                        # Read summaries one by one to handle schema differences
                        # But don't keep them all in memory - process immediately
                        all_summaries_list = []
                        for summary_file in summary_files:
                            summary = pl.read_parquet(summary_file)
                            all_summaries_list.append(summary)
                            del summary
                        
                        # Combine with diagonal_relaxed to handle schema differences
                        combined = pl.concat(all_summaries_list, how="diagonal_relaxed")
                        del all_summaries_list
                        gc.collect()
                        
                        # Sort by cells_with_age_years (descending), putting datasets with more age data first
                        # Fill null values with 0 for sorting
                        if 'cells_with_age_years' in combined.columns:
                            combined = combined.with_columns(
                                pl.col('cells_with_age_years').fill_null(0).alias('_sort_age')
                            ).sort('_sort_age', descending=True).drop('_sort_age')
                        
                        # Group by organism and save separate files
                        organisms = combined['organism'].unique().to_list()
                        
                        for organism in organisms:
                            if organism and organism != 'unknown':
                                organism_summaries = combined.filter(pl.col('organism') == organism)
                                organism_summaries = _mask_age_months_for_non_mouse(organism_summaries)
                                
                                # Create filename: replace spaces with underscores, lowercase
                                safe_organism = organism.lower().replace(' ', '_').replace('-', '_')
                                organism_file = output_dir / f"{safe_organism}_summary.{summary_format_lower}"
                                
                                # Ensure directory exists
                                organism_file.parent.mkdir(parents=True, exist_ok=True)
                                
                                # Write summary file using lazy API
                                organism_summaries.lazy().sink_csv(organism_file, separator=separator)
                                
                                summary_action.log(
                                    message_type="species_summary_created",
                                    organism=organism,
                                    file=str(organism_file),
                                    datasets=len(organism_summaries),
                                    format=summary_format_lower
                                )
                                typer.echo(f"  Generated {safe_organism}_summary.{summary_format_lower} ({len(organism_summaries)} datasets)")
                                
                                # Clean up organism-specific dataframe
                                del organism_summaries
                        
                        # Also save a combined summary with all organisms
                        all_summary_file = output_dir / f"all_datasets_summary.{summary_format_lower}"
                        all_summary_file.parent.mkdir(parents=True, exist_ok=True)
                        combined.lazy().sink_csv(all_summary_file, separator=separator)
                        summary_action.log(
                            message_type="combined_summary_created",
                            file=str(all_summary_file),
                            datasets=len(combined),
                            format=summary_format_lower
                        )
                        typer.echo(f"  Generated all_datasets_summary.{summary_format_lower} ({len(combined)} datasets)")
                        
                        # Clean up temp summaries directory
                        if summary_temp_dir.exists():
                            shutil.rmtree(summary_temp_dir)
                            summary_action.log(message_type="temp_summaries_cleaned")
                        
                        # Clean up combined dataframe
                        del combined
                        gc.collect()
        
        typer.echo(f"\n{'='*60}")
        typer.echo(f"Batch processing complete:")
        typer.echo(f"  Total files: {len(h5ad_files)}")
        typer.echo(f"  Successful: {successful}")
        typer.echo(f"  Failed: {failed}")
        typer.echo(f"  Output directory: {output_dir}")
        
        if failed > 0:
            typer.echo("\nFailed files:")
            for name, success, message in results:
                if not success and message != "skipped (exists)":
                    typer.echo(f"  - {name}: {message}")


if __name__ == "__main__":
    app()


