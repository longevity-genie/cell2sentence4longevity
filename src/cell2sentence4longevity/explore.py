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
)

app = typer.Typer(help="Extract metadata fields from h5ad AnnData files")


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
    generate_summary: bool = False
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
        
        if missing_fields:
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
        action.log(
            message_type="starting_chunk_processing",
            total_chunks=n_chunks,
            chunk_size=chunk_size
        )
        
        # Create temp directory for chunks
        temp_dir = output_path.parent / f".temp_{output_path.stem}"
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        categorical_cache: dict[str, Any] = {}
        
        for chunk_idx in tqdm(range(n_chunks), desc='Extracting metadata'):
            with start_action(action_type="process_chunk", chunk_idx=chunk_idx) as chunk_action:
                start_idx = chunk_idx * chunk_size
                end_idx = min(start_idx + chunk_size, n_cells)
                
                chunk_df = build_obs_chunk_dataframe(
                    obs_group=obs_group,
                    fields=valid_fields,
                    start_idx=start_idx,
                    end_idx=end_idx,
                    categorical_cache=categorical_cache,
                    string_fields=string_fields,
                    string_fill_value=None
                )
                
                cast_exprs = [
                    pl.col(col).cast(dtype, strict=False)
                    for col, dtype in obs_schema.items()
                    if col in chunk_df.columns
                ]
                if cast_exprs:
                    chunk_df = chunk_df.with_columns(cast_exprs)
                fill_exprs = [
                    pl.col(col).fill_null("nan")
                    for col, dtype in obs_schema.items()
                    if dtype == pl.String and col in chunk_df.columns
                ]
                if fill_exprs:
                    chunk_df = chunk_df.with_columns(fill_exprs)
                
                if extract_age:
                    chunk_df = extract_age_columns(
                        chunk_df,
                        development_stage_col=age_source_col
                    )
                
                # Write chunk to temp file
                chunk_file = temp_dir / f"chunk_{chunk_idx:04d}.parquet"
                chunk_df.write_parquet(
                    chunk_file,
                    compression=compression,
                    compression_level=compression_level,
                    use_pyarrow=use_pyarrow
                )
                
                # Clean up chunk data immediately
                del chunk_df
                
                chunk_action.log(
                    message_type="chunk_processed",
                    rows=end_idx - start_idx,
                    start_idx=start_idx,
                    end_idx=end_idx
                )
        
        # Read all chunks lazily without materializing full dataset
        action.log(message_type="building_lazy_dataframe", n_chunks=n_chunks)
        final_lazy_df = pl.scan_parquet(temp_dir / "*.parquet")
        schema_names = set(final_lazy_df.collect_schema().names())
        
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
                    action.log(
                        message_type="dataset_id_added",
                        dataset_id=dataset_id
                    )
                
                # Always attempt to join with collections (left join will add null values if not found)
                # This ensures consistent schema across all summaries
                action.log(
                    message_type="joining_with_collections",
                    dataset_id=dataset_id
                )
                final_lazy_df = join_with_collections(final_lazy_df)
                schema_names = set(final_lazy_df.collect_schema().names())
                
                # Check if dataset was actually found in collections
                if dataset_id_exists_in_collections(dataset_id):
                    action.log(
                        message_type="collection_found",
                        dataset_id=dataset_id,
                        note="Dataset found in collections, metadata added"
                    )
                else:
                    action.log(
                        message_type="dataset_not_in_collections",
                        dataset_id=dataset_id,
                        note="Dataset not found in collections cache, collection columns will be null"
                    )
            else:
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
            action.log(message_type="temp_dir_cleaned", temp_dir=str(temp_dir))
        
        written_lazy_df = pl.scan_parquet(output_path)
        written_schema = written_lazy_df.schema
        
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
            summary_schema = summary_lazy.schema
            
            def _first_non_null(column: str) -> Any | None:
                if column not in summary_schema:
                    return None
                column_type = summary_schema[column]
                if column_type == pl.Null:
                    return None
                result = (
                    summary_lazy
                    .select(pl.col(column))
                    .filter(pl.col(column).is_not_null())
                    .limit(1)
                    .collect(streaming=True)
                )
                if result.height == 0:
                    return None
                return result[column][0]
            
            with start_action(action_type="generate_summary", summary_path=str(summary_path)) as summary_action:
                summary_data: dict[str, Any] = {}
                
                dataset_id_value = _first_non_null('dataset_id')
                summary_data["dataset_id"] = dataset_id_value
                if dataset_id_value is not None:
                    summary_action.log(message_type="dataset_id_in_summary", dataset_id=dataset_id_value)
                
                summary_data["total_cells"] = n_cells
                
                organism_value = _first_non_null('organism')
                if organism_value is not None:
                    summary_data["organism"] = organism_value
                
                # Count unique donors (look for common donor column names)
                donor_cols = ['donor_id', 'donor', 'subject_id', 'individual']
                donor_col = next((col for col in donor_cols if col in summary_schema), None)
                if donor_col:
                    unique_donors_df = summary_lazy.select(
                        pl.col(donor_col).n_unique().alias('unique_donors')
                    ).collect(streaming=True)
                    unique_donors = int(unique_donors_df['unique_donors'][0])
                    summary_data["unique_donors"] = unique_donors
                    summary_action.log(message_type="donor_count", column=donor_col, count=unique_donors)
                
                # Unique tissues with counts
                tissue_cols = ['tissue', 'tissue_type', 'organ', 'tissue_general']
                tissue_col = next((col for col in tissue_cols if col in summary_schema), None)
                if tissue_col:
                    tissue_counts = (
                        summary_lazy
                        .group_by(tissue_col)
                        .agg(pl.len().alias('count'))
                        .sort('count', descending=True)
                        .collect(streaming=True)
                    )
                    unique_tissues = tissue_counts.height
                    summary_data["unique_tissues"] = unique_tissues
                    tissue_list = [
                        f"{row[tissue_col]}: {row['count']}"
                        for row in tissue_counts.iter_rows(named=True)
                        if row[tissue_col] is not None
                    ]
                    summary_data["tissues"] = ", ".join(tissue_list)
                    summary_action.log(message_type="tissue_info", column=tissue_col, count=unique_tissues)
                
                # Unique cell types with counts
                cell_type_cols = ['cell_type', 'cell_type_ontology_term_id', 'celltype']
                cell_type_col = next((col for col in cell_type_cols if col in summary_schema), None)
                if cell_type_col:
                    cell_type_counts = (
                        summary_lazy
                        .group_by(cell_type_col)
                        .agg(pl.len().alias('count'))
                        .sort('count', descending=True)
                        .collect(streaming=True)
                    )
                    unique_cell_types = cell_type_counts.height
                    summary_data["unique_cell_types"] = unique_cell_types
                    cell_type_list = [
                        f"{row[cell_type_col]}: {row['count']}"
                        for row in cell_type_counts.iter_rows(named=True)
                        if row[cell_type_col] is not None
                    ]
                    summary_data["cell_types"] = ", ".join(cell_type_list)
                    summary_action.log(message_type="cell_type_info", column=cell_type_col, count=unique_cell_types)
                
                # Assay/Technology information
                assay_cols = ['assay', 'assay_ontology_term_id', 'technology', 'platform']
                assay_col = next((col for col in assay_cols if col in summary_schema), None)
                if assay_col:
                    assay_counts = (
                        summary_lazy
                        .group_by(assay_col)
                        .agg(pl.len().alias('count'))
                        .sort('count', descending=True)
                        .collect(streaming=True)
                    )
                    unique_assays = assay_counts.height
                    summary_data["unique_assays"] = unique_assays
                    assay_list = [
                        f"{row[assay_col]}: {row['count']}"
                        for row in assay_counts.iter_rows(named=True)
                        if row[assay_col] is not None
                    ]
                    summary_data["assays"] = ", ".join(assay_list)
                    summary_action.log(message_type="assay_info", column=assay_col, count=unique_assays)
                
                # Sex information
                sex_cols = ['sex', 'sex_ontology_term_id', 'gender']
                sex_col = next((col for col in sex_cols if col in summary_schema), None)
                if sex_col:
                    sex_counts = (
                        summary_lazy
                        .group_by(sex_col)
                        .agg(pl.len().alias('count'))
                        .sort('count', descending=True)
                        .collect(streaming=True)
                    )
                    unique_sexes = sex_counts.height
                    summary_data["unique_sexes"] = unique_sexes
                    sex_list = [
                        f"{row[sex_col]}: {row['count']}"
                        for row in sex_counts.iter_rows(named=True)
                        if row[sex_col] is not None
                    ]
                    summary_data["sexes"] = ", ".join(sex_list)
                    summary_action.log(message_type="sex_info", column=sex_col, count=unique_sexes)
                
                # Disease information
                disease_cols = ['disease', 'disease_ontology_term_id', 'condition']
                disease_col = next((col for col in disease_cols if col in summary_schema), None)
                if disease_col:
                    disease_counts = (
                        summary_lazy
                        .group_by(disease_col)
                        .agg(pl.len().alias('count'))
                        .sort('count', descending=True)
                        .collect(streaming=True)
                    )
                    unique_diseases = disease_counts.height
                    summary_data["unique_diseases"] = unique_diseases
                    disease_list = [
                        f"{row[disease_col]}: {row['count']}"
                        for row in disease_counts.iter_rows(named=True)
                        if row[disease_col] is not None
                    ]
                    summary_data["diseases"] = ", ".join(disease_list)
                    summary_action.log(message_type="disease_info", column=disease_col, count=unique_diseases)
                
                # Age statistics (if age columns were extracted)
                if 'age_years' in summary_schema:
                    age_years_non_null = summary_lazy.filter(pl.col('age_years').is_not_null())
                    age_years_stats = (
                        age_years_non_null
                        .select([
                            pl.col('age_years').min().alias('min_age'),
                            pl.col('age_years').max().alias('max_age'),
                            pl.col('age_years').mean().alias('mean_age'),
                            pl.len().alias('count')
                        ])
                        .collect(streaming=True)
                    )
                    count_years = int(age_years_stats['count'][0])
                    if count_years > 0:
                        min_age = age_years_stats['min_age'][0]
                        max_age = age_years_stats['max_age'][0]
                        mean_age = round(age_years_stats['mean_age'][0], 2)
                        unique_ages_df = (
                            age_years_non_null
                            .select(pl.col('age_years'))
                            .unique()
                            .sort('age_years')
                            .collect(streaming=True)
                        )
                        unique_ages = [val for val in unique_ages_df['age_years'].to_list() if val is not None]
                        summary_data["age_years_min"] = min_age
                        summary_data["age_years_max"] = max_age
                        summary_data["age_years_mean"] = mean_age
                        summary_data["unique_ages_years"] = ", ".join(str(age) for age in unique_ages)
                        summary_data["cells_with_age_years"] = count_years
                        summary_action.log(
                            message_type="age_years_stats",
                            min=min_age,
                            max=max_age,
                            mean=mean_age,
                            unique_ages=len(unique_ages),
                            count=count_years
                        )
                
                if 'age_months' in summary_schema:
                    age_months_non_null = summary_lazy.filter(pl.col('age_months').is_not_null())
                    age_months_stats = (
                        age_months_non_null
                        .select([
                            pl.col('age_months').min().alias('min_age'),
                            pl.col('age_months').max().alias('max_age'),
                            pl.col('age_months').mean().alias('mean_age'),
                            pl.len().alias('count')
                        ])
                        .collect(streaming=True)
                    )
                    count_months = int(age_months_stats['count'][0])
                    if count_months > 0:
                        min_age_m = age_months_stats['min_age'][0]
                        max_age_m = age_months_stats['max_age'][0]
                        mean_age_m = round(age_months_stats['mean_age'][0], 2)
                        unique_ages_m_df = (
                            age_months_non_null
                            .select(pl.col('age_months'))
                            .unique()
                            .sort('age_months')
                            .collect(streaming=True)
                        )
                        unique_ages_m = [val for val in unique_ages_m_df['age_months'].to_list() if val is not None]
                        summary_data["age_months_min"] = min_age_m
                        summary_data["age_months_max"] = max_age_m
                        summary_data["age_months_mean"] = mean_age_m
                        summary_data["unique_ages_months"] = ", ".join(str(age) for age in unique_ages_m)
                        summary_data["cells_with_age_months"] = count_months
                        summary_action.log(
                            message_type="age_months_stats",
                            min=min_age_m,
                            max=max_age_m,
                            mean=mean_age_m,
                            unique_ages=len(unique_ages_m),
                            count=count_months
                        )
                
                # Collection info at the end (if available)
                collection_id_value = _first_non_null('collection_id')
                summary_data["collection_id"] = collection_id_value
                
                publication_title = _first_non_null('publication_title')
                summary_data["publication_title"] = publication_title
                
                publication_doi = _first_non_null('publication_doi')
                summary_data["publication_doi"] = publication_doi
                
                # Create and save summary dataframe (single row with columns)
                summary_df = pl.DataFrame([summary_data])
                summary_df.write_csv(summary_path)
                
                summary_action.log(
                    message_type="summary_complete",
                    path=str(summary_path),
                    columns_count=len(summary_df.columns)
                )
        
        # Close the h5ad file and clean up
        adata.file.close()
        del adata, final_lazy_df, written_lazy_df
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
    else:
        # Default to ./logs directory
        default_log_dir = Path("./logs")
        default_log_dir.mkdir(parents=True, exist_ok=True)
        json_log = default_log_dir / "extract.json"
        rendered_log = default_log_dir / "extract.log"
        to_nice_file(output_file=json_log, rendered_file=rendered_log)
    
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
    input_dir: Path = typer.Argument(..., help="Directory containing h5ad files"),
    output_dir: Optional[Path] = typer.Option(None, "--output-dir", "-o", help="Output directory for parquet files (default: data/output/meta)"),
    fields: Optional[list[str]] = typer.Option(None, "--field", "-f", help="Field name to extract (can be specified multiple times). If not specified, extracts all available metadata fields."),
    chunk_size: int = typer.Option(10000, "--chunk-size", help="Number of rows to process at a time"),
    compression: str = typer.Option("zstd", "--compression", help="Compression algorithm"),
    compression_level: int = typer.Option(3, "--compression-level", help="Compression level"),
    extract_age: bool = typer.Option(True, "--extract-age/--no-extract-age", help="Extract age from development_stage column"),
    age_source_col: str = typer.Option("development_stage", "--age-source", help="Column to extract age from"),
    generate_summary: bool = typer.Option(False, "--summary/--no-summary", help="Generate species-level summary files"),
    summary_format: str = typer.Option("csv", "--summary-format", help="Format for summary files: 'csv' or 'tsv' (default: csv)"),
    log_dir: Optional[Path] = typer.Option(None, "--log-dir", help="Directory for log files"),
    skip_existing: bool = typer.Option(True, "--skip-existing/--overwrite", help="Skip files that already have output"),
    max_threads: Optional[int] = typer.Option(None, "--max-threads", help="Maximum number of threads for Polars (default: 8, set to 1 for minimal memory)"),
    max_file_size_mb: Optional[float] = typer.Option(None, "--max-file-size-mb", help="Maximum file size in MB to process (e.g., 12000 for 12 GB). Files larger than this will be skipped."),
) -> None:
    """Extract metadata fields from multiple h5ad files in batch.
    
    By default, extracts all available metadata fields from adata.obs.
    Specify -f/--field to extract only specific fields.
    
    Examples:
        # Extract all metadata fields from all h5ad files to default location
        explore batch ./data/input
        
        # Extract specific fields only
        explore batch ./data/input -f age -f cell_type -f tissue
        
        # Specify custom output directory
        explore batch ./data/input --output-dir ./custom/output
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
    
    # Determine output directory
    if output_dir is None:
        output_dir = Path("data/output/meta")
    
    # Setup logging
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
        json_log = log_dir / "batch_extract.json"
        rendered_log = log_dir / "batch_extract.log"
        to_nice_file(output_file=json_log, rendered_file=rendered_log)
    else:
        # Default to ./logs directory
        default_log_dir = Path("./logs")
        default_log_dir.mkdir(parents=True, exist_ok=True)
        json_log = default_log_dir / "batch_extract.json"
        rendered_log = default_log_dir / "batch_extract.log"
        to_nice_file(output_file=json_log, rendered_file=rendered_log)
    
    with start_action(
        action_type="batch_extract_fields",
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        fields=fields if fields else "all_fields"
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
                        generate_summary=False  # Never generate individual summaries in batch mode
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
                        h5ad_file = input_dir / f"{dataset_id}.h5ad"
                        
                        try:
                            # Read metadata (only what we need)
                            meta_df = pl.scan_parquet(meta_file).collect()
                            
                            # Get organism
                            organism = None
                            if 'organism' in meta_df.columns:
                                organism = meta_df['organism'][0]
                            elif h5ad_file.exists():
                                # Try to get from h5ad file
                                try:
                                    adata = ad.read_h5ad(h5ad_file, backed='r')
                                    if 'organism' in adata.obs.columns:
                                        organism = adata.obs['organism'][0]
                                    adata.file.close()
                                    del adata
                                except Exception as e:
                                    summary_action.log(
                                        message_type="error_reading_h5ad",
                                        file=h5ad_file.name,
                                        error=str(e)
                                    )
                            
                            if organism is None:
                                organism = 'Homo sapiens'  # Default to human if not specified
                                summary_action.log(
                                    message_type="organism_default_to_human",
                                    dataset_id=dataset_id
                                )
                            
                            # Create summary from metadata (same logic as in extract_fields_from_h5ad)
                            summary_data: dict[str, any] = {}
                            
                            # Dataset ID
                            summary_data["dataset_id"] = dataset_id
                            
                            # Organism (put early in the output)
                            summary_data["organism"] = organism
                            
                            # Total cell count
                            summary_data["total_cells"] = len(meta_df)
                            
                            # Unique donors
                            donor_cols = ['donor_id', 'donor', 'subject_id', 'individual']
                            donor_col = next((col for col in donor_cols if col in meta_df.columns), None)
                            if donor_col:
                                summary_data["unique_donors"] = meta_df[donor_col].n_unique()
                            
                            # Tissues with counts
                            tissue_cols = ['tissue', 'tissue_type', 'organ', 'tissue_general']
                            tissue_col = next((col for col in tissue_cols if col in meta_df.columns), None)
                            if tissue_col:
                                tissue_counts = meta_df.group_by(tissue_col).agg(pl.len().alias('count')).sort('count', descending=True)
                                summary_data["unique_tissues"] = tissue_counts.height
                                tissue_list = [f"{row[tissue_col]}: {row['count']}" for row in tissue_counts.iter_rows(named=True) if row[tissue_col] is not None]
                                summary_data["tissues"] = ", ".join(tissue_list)
                            
                            # Cell types with counts
                            cell_type_cols = ['cell_type', 'cell_type_ontology_term_id', 'celltype']
                            cell_type_col = next((col for col in cell_type_cols if col in meta_df.columns), None)
                            if cell_type_col:
                                cell_type_counts = meta_df.group_by(cell_type_col).agg(pl.len().alias('count')).sort('count', descending=True)
                                summary_data["unique_cell_types"] = cell_type_counts.height
                                cell_type_list = [f"{row[cell_type_col]}: {row['count']}" for row in cell_type_counts.iter_rows(named=True) if row[cell_type_col] is not None]
                                summary_data["cell_types"] = ", ".join(cell_type_list)
                            
                            # Assays with counts
                            assay_cols = ['assay', 'assay_ontology_term_id', 'technology', 'platform']
                            assay_col = next((col for col in assay_cols if col in meta_df.columns), None)
                            if assay_col:
                                assay_counts = meta_df.group_by(assay_col).agg(pl.len().alias('count')).sort('count', descending=True)
                                summary_data["unique_assays"] = assay_counts.height
                                assay_list = [f"{row[assay_col]}: {row['count']}" for row in assay_counts.iter_rows(named=True) if row[assay_col] is not None]
                                summary_data["assays"] = ", ".join(assay_list)
                            
                            # Sex with counts
                            sex_cols = ['sex', 'sex_ontology_term_id', 'gender']
                            sex_col = next((col for col in sex_cols if col in meta_df.columns), None)
                            if sex_col:
                                sex_counts = meta_df.group_by(sex_col).agg(pl.len().alias('count')).sort('count', descending=True)
                                summary_data["unique_sexes"] = sex_counts.height
                                sex_list = [f"{row[sex_col]}: {row['count']}" for row in sex_counts.iter_rows(named=True) if row[sex_col] is not None]
                                summary_data["sexes"] = ", ".join(sex_list)
                            
                            # Diseases with counts
                            disease_cols = ['disease', 'disease_ontology_term_id', 'condition']
                            disease_col = next((col for col in disease_cols if col in meta_df.columns), None)
                            if disease_col:
                                disease_counts = meta_df.group_by(disease_col).agg(pl.len().alias('count')).sort('count', descending=True)
                                summary_data["unique_diseases"] = disease_counts.height
                                disease_list = [f"{row[disease_col]}: {row['count']}" for row in disease_counts.iter_rows(named=True) if row[disease_col] is not None]
                                summary_data["diseases"] = ", ".join(disease_list)
                            
                            # Age statistics
                            if 'age_years' in meta_df.columns:
                                age_years_data = meta_df.filter(pl.col('age_years').is_not_null())
                                if len(age_years_data) > 0:
                                    summary_data["age_years_min"] = age_years_data['age_years'].min()
                                    summary_data["age_years_max"] = age_years_data['age_years'].max()
                                    summary_data["age_years_mean"] = round(age_years_data['age_years'].mean(), 2)
                                    unique_ages = sorted(age_years_data['age_years'].unique().to_list())
                                    summary_data["unique_ages_years"] = ", ".join(str(age) for age in unique_ages)
                                    summary_data["cells_with_age_years"] = len(age_years_data)
                            
                            if 'age_months' in meta_df.columns:
                                age_months_data = meta_df.filter(pl.col('age_months').is_not_null())
                                if len(age_months_data) > 0:
                                    summary_data["age_months_min"] = age_months_data['age_months'].min()
                                    summary_data["age_months_max"] = age_months_data['age_months'].max()
                                    summary_data["age_months_mean"] = round(age_months_data['age_months'].mean(), 2)
                                    unique_ages_m = sorted(age_months_data['age_months'].unique().to_list())
                                    summary_data["unique_ages_months"] = ", ".join(str(age) for age in unique_ages_m)
                                    summary_data["cells_with_age_months"] = len(age_months_data)
                            
                            # Collection info at the end (if available)
                            if 'collection_id' in meta_df.columns:
                                summary_data["collection_id"] = meta_df['collection_id'][0]
                            if 'publication_title' in meta_df.columns:
                                summary_data["publication_title"] = meta_df['publication_title'][0]
                            if 'publication_doi' in meta_df.columns:
                                summary_data["publication_doi"] = meta_df['publication_doi'][0]
                            
                            # Create summary dataframe and write immediately
                            summary_df = pl.DataFrame([summary_data])
                            summary_file = summary_temp_dir / f"summary_{idx:04d}.parquet"
                            summary_df.write_parquet(summary_file)
                            
                            # Clean up
                            del meta_df, summary_df, summary_data
                            
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


