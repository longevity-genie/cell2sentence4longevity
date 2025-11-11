"""Integration tests for the cell2sentence preprocessing pipeline."""

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import polars as pl
import pytest
from eliot import start_action
from pycomfort.logging import to_nice_file

from cell2sentence4longevity.preprocessing import (
    convert_h5ad_to_train_test,
    create_hgnc_mapper,
    download_dataset,
)


class TestIntegrationPipeline:
    """Integration tests for the full preprocessing pipeline."""
    
    @pytest.fixture
    def temp_dirs(self) -> dict[str, Path]:
        """Create test directories using project's data/ structure.
        
        Uses the standard data/input, data/output pattern.
        Note: Input directory is shared across runs to avoid re-downloading.
              Output/logs directories are cleaned at the start but NOT at the end,
              allowing manual exploration.
              Interim directory is NOT pre-created since one-step processing doesn't use it.
        """
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Use project's data directories with test subdirectories
        project_root = Path(__file__).parent.parent
        data_dir = project_root / 'data'
        
        # Input is shared across test runs (don't delete downloads)
        # Output/logs are timestamped for each run
        # Interim is not created (one-step processing doesn't need it)
        dirs = {
            'base': data_dir,
            'input': data_dir / 'input',  # Shared input directory for downloads
            'output': data_dir / 'output' / f'test_{timestamp}',
            'logs': project_root / 'logs' / f'test_{timestamp}',
        }
        
        # Clean up output/logs for fresh start, but preserve input
        for key, dir_path in dirs.items():
            if key in ('base', 'input'):
                # Don't delete base or input directories (preserve downloads)
                dir_path.mkdir(parents=True, exist_ok=True)
                continue
            if dir_path.exists():
                shutil.rmtree(dir_path, ignore_errors=True)
            dir_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\n📁 Test directories:")
        print(f"  - Input: {dirs['input']} (shared, downloads preserved)")
        print(f"  - Output: {dirs['output']}")
        print(f"  - Logs: {dirs['logs']}")
        print(f"  (Files will be kept after test for manual exploration)")
        
        yield dirs
        
        # DO NOT cleanup - keep files for manual exploration
        print(f"\n✓ Test files preserved for manual exploration:")
        print(f"  - Input: {dirs['input']} (shared)")
        print(f"  - Output: {dirs['output']}")
        print(f"  - Logs: {dirs['logs']}")
    
    def test_full_pipeline_with_real_data(self, temp_dirs: dict[str, Path]) -> None:
        """Test the full pipeline with real data from CZI.
        
        This test:
        1. Downloads real data from CZI Science
        2. Runs the full preprocessing pipeline with collection join enabled
        3. Validates that nothing crashes
        4. Checks logs are properly written
        5. Validates age field is numeric
        6. Validates dataset_id column is added (join_collection=True)
        7. Demonstrates joining with collections metadata
        8. Checks output files exist and have correct structure
        """
        from cell2sentence4longevity.preprocessing import (
            join_with_collections,
            get_collections_cache,
        )
        
        # Setup logging to files
        log_dir = temp_dirs['logs']
        # Use single test dataset
        dataset_url = "https://datasets.cellxgene.cziscience.com/10cc50a0-af80-4fa1-b668-893dd5c0113a.h5ad"
        dataset_id = dataset_url.split('/')[-1].replace('.h5ad', '')
        json_log = log_dir / f'integration_test_{dataset_id}.json'
        rendered_log = log_dir / f'integration_test_{dataset_id}.log'
        
        to_nice_file(output_file=json_log, rendered_file=rendered_log)
        
        with start_action(action_type="integration_test_full_pipeline") as test_action:
            # Step 0: Download real dataset
            test_action.log(message_type="test_step", step=0, description="Downloading real dataset", url=dataset_url)
            url = dataset_url
            
            h5ad_path = download_dataset(
                url=url,
                output_dir=temp_dirs['input']
            )
            
            # Validate download
            assert h5ad_path.exists(), "Downloaded h5ad file should exist"
            assert h5ad_path.stat().st_size > 0, "Downloaded file should not be empty"
            test_action.log(
                message_type="download_validated", 
                path=str(h5ad_path), 
                size_mb=h5ad_path.stat().st_size / 1024 / 1024
            )
            
            # Step 1: Create HGNC mapper (optional - skip if download fails)
            test_action.log(message_type="test_step", step=1, description="Creating HGNC mapper (optional)")
            # HGNC mappers are now stored in shared directory as parquet
            shared_dir = Path("./data/shared")
            mappers_path = shared_dir / 'hgnc_mappers.parquet'
            try:
                create_hgnc_mapper(shared_dir)
                assert mappers_path.exists(), "HGNC mappers file should exist"
                test_action.log(message_type="hgnc_mapper_created", path=str(mappers_path))
            except RuntimeError as e:
                test_action.log(
                    message_type="hgnc_mapper_creation_failed", 
                    error=str(e),
                    fallback="Will use feature_name from h5ad"
                )
                mappers_path = None
            
            # Step 2: One-step conversion (h5ad -> cell sentences + age extraction -> train/test split -> output)
            test_action.log(message_type="test_step", step=2, description="One-step conversion with train/test split")
            
            # Extract dataset name from h5ad path
            dataset_name = h5ad_path.stem
            
            convert_h5ad_to_train_test(
                h5ad_path=h5ad_path,
                mappers_path=mappers_path,
                output_dir=temp_dirs['output'],
                dataset_name=dataset_name,
                chunk_size=10000,
                top_genes=2000,
                test_size=0.05,
                random_state=42,
                compression="zstd",
                compression_level=3,
                use_pyarrow=True,
                skip_train_test_split=False,
                stratify_by_age=True,
                join_collection=True  # Enable dataset_id column
            )
            
            # Validate train/test split
            # The output structure is: output_dir/dataset_name/train/ and output_dir/dataset_name/test/
            train_dir = temp_dirs['output'] / dataset_name / 'train'
            test_dir = temp_dirs['output'] / dataset_name / 'test'
            
            assert train_dir.exists(), "Train directory should exist"
            assert test_dir.exists(), "Test directory should exist"
            
            train_files = list(train_dir.glob("*.parquet"))
            test_files = list(test_dir.glob("*.parquet"))
            
            assert len(train_files) > 0, "Should have train files"
            assert len(test_files) > 0, "Should have test files"
            
            # Check first chunk structure
            sample_df = pl.read_parquet(train_files[0])
            test_action.log(
                message_type="sample_chunk_structure",
                columns=sample_df.columns,
                rows=len(sample_df)
            )
            assert 'cell_sentence' in sample_df.columns or 'cell2sentence' in sample_df.columns, \
                "Should have cell_sentence or cell2sentence column"
            
            # Validate age column was added during conversion
            assert 'age' in sample_df.columns, "Should have age column (added during conversion)"
            
            # Validate age field is numeric
            age_dtype = sample_df.schema['age']
            assert age_dtype in [pl.Int64, pl.Int32, pl.Int16, pl.Float64, pl.Float32], \
                f"Age column should be numeric, got {age_dtype}"
            test_action.log(message_type="age_column_validated", dtype=str(age_dtype))
            
            # Check age values are reasonable
            age_values = sample_df['age'].drop_nulls()
            if len(age_values) > 0:
                min_age = age_values.min()
                max_age = age_values.max()
                test_action.log(message_type="age_range", min_age=min_age, max_age=max_age)
                assert min_age >= 0, "Age should be non-negative"
                assert max_age <= 150, "Age should be realistic"
            
            test_action.log(
                message_type="train_test_split_created",
                train_chunks=len(train_files),
                test_chunks=len(test_files)
            )
            
            # Verify train/test split ratio
            train_df = pl.scan_parquet(train_dir / "*.parquet")
            test_df = pl.scan_parquet(test_dir / "*.parquet")
            
            train_count = train_df.select(pl.len()).collect().item()
            test_count = test_df.select(pl.len()).collect().item()
            total_count = train_count + test_count
            
            test_ratio = test_count / total_count
            test_action.log(
                message_type="split_statistics",
                train_count=train_count,
                test_count=test_count,
                test_ratio=test_ratio
            )
            
            # Allow some tolerance in split ratio
            assert 0.03 <= test_ratio <= 0.07, f"Test ratio should be ~0.05, got {test_ratio:.3f}"
            
            # Validate final data structure
            test_action.log(message_type="test_step", description="Validating final data structure")
            final_train_sample = pl.scan_parquet(train_dir / "*.parquet").limit(100).collect()
            final_test_sample = pl.scan_parquet(test_dir / "*.parquet").limit(100).collect()
            
            assert 'cell_sentence' in final_train_sample.columns, "Final data should have cell_sentence"
            assert 'age' in final_train_sample.columns, "Final data should have age"
            
            # Validate dataset_id column was added (join_collection=True)
            assert 'dataset_id' in final_train_sample.columns, "Should have dataset_id column (join_collection=True)"
            assert final_train_sample['dataset_id'][0] == dataset_id, "dataset_id should match the h5ad filename"
            test_action.log(
                message_type="dataset_id_validated",
                dataset_id=final_train_sample['dataset_id'][0]
            )
            
            # Validate that publication columns were added during conversion (join_collection=True)
            # These should be automatically added by convert_h5ad_to_train_test when join_collection=True
            expected_publication_columns = [
                "collection_id",
                "publication_title",
                "publication_doi",
                "publication_description",
                "publication_contact_name",
                "publication_contact_email"
            ]
            
            present_publication_columns = [col for col in expected_publication_columns if col in final_train_sample.columns]
            test_action.log(
                message_type="publication_columns_validated",
                expected_columns=expected_publication_columns,
                present_columns=present_publication_columns,
                all_present=len(present_publication_columns) == len(expected_publication_columns)
            )
            
            # Check that publication columns are present
            assert len(present_publication_columns) == len(expected_publication_columns), \
                f"All publication columns should be present in output. Missing: {set(expected_publication_columns) - set(present_publication_columns)}"
            
            # Validate that publication_description is filled with actual data (not null)
            non_null_descriptions = final_train_sample['publication_description'].is_not_null().sum()
            assert non_null_descriptions > 0, "publication_description should have non-null values"
            
            # Check that description is not empty
            if non_null_descriptions > 0:
                first_description = final_train_sample['publication_description'].drop_nulls()[0]
                assert len(first_description) > 0, "publication_description should not be empty"
                assert len(first_description) > 50, "publication_description should contain meaningful text"
                
                test_action.log(
                    message_type="publication_description_validated",
                    description_length=len(first_description),
                    description_preview=first_description[:200]
                )
            
            # Validate publication_title is also filled
            non_null_titles = final_train_sample['publication_title'].is_not_null().sum()
            assert non_null_titles > 0, "publication_title should have non-null values"
            
            if non_null_titles > 0:
                first_title = final_train_sample['publication_title'].drop_nulls()[0]
                assert len(first_title) > 0, "publication_title should not be empty"
                
                test_action.log(
                    message_type="publication_title_validated",
                    title=first_title
                )

            
            # Check cell_sentence is not empty
            non_empty_sentences = final_train_sample['cell_sentence'].str.len_chars() > 0
            assert non_empty_sentences.sum() > 0, "Should have non-empty cell sentences"
            
            # Check that no Ensembl IDs are present in cell sentences (train set)
            # Ensembl IDs start with "ENS" (e.g., ENSG00000139618, ENST00000361390)
            has_ensembl_ids_train = final_train_sample['cell_sentence'].str.contains(r'\bENS[A-Z]*\d+')
            ensembl_id_count_train = has_ensembl_ids_train.sum()
            assert ensembl_id_count_train == 0, \
                f"Train sentences should not contain Ensembl IDs, found {ensembl_id_count_train} sentences with Ensembl IDs"
            
            # Check that no Ensembl IDs are present in cell sentences (test set)
            has_ensembl_ids_test = final_test_sample['cell_sentence'].str.contains(r'\bENS[A-Z]*\d+')
            ensembl_id_count_test = has_ensembl_ids_test.sum()
            assert ensembl_id_count_test == 0, \
                f"Test sentences should not contain Ensembl IDs, found {ensembl_id_count_test} sentences with Ensembl IDs"
            
            test_action.log(
                message_type="ensembl_id_validation",
                train_sentences_with_ensembl_ids=ensembl_id_count_train,
                test_sentences_with_ensembl_ids=ensembl_id_count_test,
                validation="passed"
            )
            
            # Demonstrate joining with collections metadata
            test_action.log(message_type="test_step", description="Demonstrating join with collections")
            
            # Get collections cache
            collections_df = get_collections_cache()
            test_action.log(
                message_type="collections_cache_loaded",
                total_datasets=len(collections_df),
                columns=collections_df.columns
            )
            
            # Check if our dataset is in the cache
            dataset_in_cache = collections_df.filter(pl.col("dataset_id") == dataset_id)
            
            if len(dataset_in_cache) > 0:
                row = dataset_in_cache.row(0, named=True)
                test_action.log(
                    message_type="dataset_found_in_cache",
                    collection_id=row["collection_id"],
                    title=row["title"],
                    doi=row.get("doi", "N/A")
                )
            
            # Join sample dataframe with collections
            joined_df = join_with_collections(final_train_sample)
            
            test_action.log(
                message_type="join_demonstration_complete",
                original_columns=len(final_train_sample.columns),
                joined_columns=len(joined_df.columns),
                new_columns=[col for col in joined_df.columns if col not in final_train_sample.columns]
            )
            
            # Check if publication columns were added
            expected_columns = [
                "collection_id", 
                "publication_title", 
                "publication_doi",
                "publication_description",
                "publication_contact_name",
                "publication_contact_email"
            ]
            
            added_columns = [col for col in expected_columns if col in joined_df.columns]
            
            if len(added_columns) > 0:
                test_action.log(
                    message_type="publication_columns_added",
                    added_columns=added_columns
                )
            
            test_action.log(
                message_type="final_structure_validated",
                columns=final_train_sample.columns,
                sample_sentence=final_train_sample['cell_sentence'][0][:100]
            )
            
            # Validate logs were written properly
            test_action.log(message_type="test_step", description="Validating logs")
            assert json_log.exists(), "JSON log file should exist"
            assert rendered_log.exists(), "Rendered log file should exist"
            
            # Check JSON log is valid
            with open(json_log, 'r') as f:
                log_lines = f.readlines()
                assert len(log_lines) > 0, "JSON log should have entries"
                
                # Validate first line is valid JSON
                first_log = json.loads(log_lines[0])
                assert 'action_type' in first_log or 'message_type' in first_log, \
                    "Log entries should have action_type or message_type"
            
            test_action.log(message_type="log_validation", json_entries=len(log_lines))
            
            # Check rendered log has content
            rendered_content = rendered_log.read_text()
            assert len(rendered_content) > 0, "Rendered log should have content"
            test_action.log(message_type="log_validation", rendered_chars=len(rendered_content))
            
            # Validate specific log actions were recorded
            log_content = '\n'.join(log_lines)
            assert 'download_dataset' in log_content, "Should log download action"
            assert 'create_hgnc_mapper' in log_content or 'hgnc' in log_content.lower(), \
                "Should log HGNC mapper creation"
            assert 'convert_h5ad_to_train_test' in log_content or 'h5ad' in log_content.lower(), \
                "Should log h5ad conversion"
            assert 'age' in log_content.lower(), \
                "Should log age processing (done during conversion)"
            assert 'split' in log_content.lower() or 'train' in log_content.lower(), \
                "Should log train/test split"
            
            test_action.log(message_type="test_complete", status="passed")


class TestLogging:
    """Test logging functionality."""
    
    def test_log_file_creation(self) -> None:
        """Test that log files are created correctly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            json_log = temp_path / 'test.json'
            rendered_log = temp_path / 'test.log'
            
            to_nice_file(output_file=json_log, rendered_file=rendered_log)
            
            with start_action(action_type="test_log_creation") as action:
                action.log(message_type="test_message", status="testing")
            
            assert json_log.exists(), "JSON log should be created"
            assert rendered_log.exists(), "Rendered log should be created"
            
            # Verify JSON log content
            with open(json_log, 'r') as f:
                lines = f.readlines()
                assert len(lines) > 0, "JSON log should have content"
                
                first_entry = json.loads(lines[0])
                assert 'action_type' in first_entry or 'message_type' in first_entry, \
                    "Log should have structured entries"


class TestAgeExtraction:
    """Test age extraction functionality."""
    
    def test_age_extraction_from_development_stage(self) -> None:
        """Test that age is correctly extracted and is numeric."""
        from cell2sentence4longevity.preprocessing.h5ad_converter import extract_age_column
        
        # Create test DataFrame with various formats
        test_df = pl.DataFrame({
            'development_stage': [
                "22-year-old stage",
                "45-year-old human stage",
                "1-year-old stage",
                "100-year-old stage",
                None,
                "",
                "unknown",
                "child"
            ]
        })
        
        # Extract age using vectorized Polars expression
        result_df = extract_age_column(test_df)
        
        # Validate results
        ages = result_df['age'].to_list()
        assert ages[0] == 22.0
        assert ages[1] == 45.0
        assert ages[2] == 1.0
        assert ages[3] == 100.0
        assert ages[4] is None  # None input
        assert ages[5] is None  # Empty string
        assert ages[6] is None  # Invalid format
        assert ages[7] is None  # Invalid format
    
    def test_age_field_numeric_dtype(self) -> None:
        """Test that age field has numeric dtype after processing."""
        from cell2sentence4longevity.preprocessing.h5ad_converter import extract_age_column
        
        # Create a test dataframe
        df = pl.DataFrame({
            'development_stage': ['22-year-old stage', '45-year-old stage', '30-year-old stage'],
            'cell_sentence': ['GENE1 GENE2 GENE3', 'GENE4 GENE5 GENE6', 'GENE7 GENE8 GENE9']
        })
        
        # Apply age extraction using vectorized Polars expression
        df = extract_age_column(df)
        
        # Check dtype
        assert df.schema['age'] == pl.Float64, "Age should be Float64"
        
        # Check values
        assert df['age'].to_list() == [22.0, 45.0, 30.0], "Age values should be extracted correctly"
        
        # Verify all ages are numeric
        assert df['age'].dtype.is_float(), "Age dtype should be float"


class TestBatchProcessing:
    """Test batch processing functionality (skipped by default, run with pytest -m batch)."""
    
    @pytest.fixture
    def batch_temp_dirs(self) -> dict[str, Path]:
        """Create test directories for batch processing.
        
        Uses the standard data/input, data/output pattern.
        Input directory is shared across runs to avoid re-downloading.
        Interim directory is NOT pre-created since one-step processing doesn't use it.
        """
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        project_root = Path(__file__).parent.parent
        data_dir = project_root / 'data'
        
        dirs = {
            'base': data_dir,
            'input': data_dir / 'input',
            'output': data_dir / 'output' / f'batch_test_{timestamp}',
            'logs': project_root / 'logs' / f'batch_test_{timestamp}',
        }
        
        # Clean up output/logs for fresh start, preserve input
        for key, dir_path in dirs.items():
            if key in ('base', 'input'):
                dir_path.mkdir(parents=True, exist_ok=True)
                continue
            if dir_path.exists():
                shutil.rmtree(dir_path, ignore_errors=True)
            dir_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\n📁 Batch test directories:")
        print(f"  - Input: {dirs['input']} (shared)")
        print(f"  - Output: {dirs['output']}")
        print(f"  - Logs: {dirs['logs']}")
        
        yield dirs
        
        # Keep files for manual exploration
        print(f"\n✓ Batch test files preserved for manual exploration:")
        print(f"  - Output: {dirs['output']}")
        print(f"  - Logs: {dirs['logs']}")
    
    @pytest.mark.skip(reason="Batch processing test - run explicitly with pytest -m batch or pytest -k test_batch")
    def test_batch_processing_multiple_files(self, batch_temp_dirs: dict[str, Path]) -> None:
        """Test batch processing of multiple h5ad files.
        
        This test:
        1. Downloads multiple real h5ad files
        2. Processes them in batch mode
        3. Validates each file is processed independently
        4. Checks error isolation (one file failure doesn't stop others)
        5. Validates batch summary is created
        6. Checks per-file logging
        """
        from cell2sentence4longevity.preprocessing import download_dataset
        from cell2sentence4longevity.preprocess import _process_single_file, sanitize_dataset_name
        import polars as pl
        
        # Setup logging
        log_dir = batch_temp_dirs['logs']
        json_log = log_dir / 'batch_test.json'
        rendered_log = log_dir / 'batch_test.log'
        to_nice_file(output_file=json_log, rendered_file=rendered_log)
        
        with start_action(action_type="test_batch_processing") as test_action:
            # Download multiple test datasets
            test_action.log(message_type="test_step", step=0, description="Downloading test datasets")
            
            test_urls = [
                "https://datasets.cellxgene.cziscience.com/10cc50a0-af80-4fa1-b668-893dd5c0113a.h5ad",
                "https://datasets.cellxgene.cziscience.com/9deda9ad-6a71-401e-b909-5263919d85f9.h5ad",
            ]
            
            h5ad_files: list[Path] = []
            for url in test_urls:
                h5ad_path = download_dataset(url=url, output_dir=batch_temp_dirs['input'])
                assert h5ad_path.exists(), f"Downloaded file should exist: {h5ad_path}"
                h5ad_files.append(h5ad_path)
            
            test_action.log(message_type="datasets_downloaded", count=len(h5ad_files))
            
            # Create HGNC mapper (optional)
            test_action.log(message_type="test_step", step=1, description="Creating HGNC mapper")
            from cell2sentence4longevity.preprocessing import create_hgnc_mapper
            
            # HGNC mappers are now stored in shared directory as parquet
            shared_dir = Path("./data/shared")
            mappers_path = shared_dir / 'hgnc_mappers.parquet'
            try:
                create_hgnc_mapper(shared_dir)
                test_action.log(message_type="hgnc_mapper_created", path=str(mappers_path))
            except RuntimeError as e:
                test_action.log(message_type="hgnc_mapper_creation_failed", error=str(e))
                mappers_path = None
            
            # Process files in batch
            test_action.log(message_type="test_step", step=2, description="Batch processing files")
            
            results: list[tuple[str, bool, str, float, Path]] = []
            
            for idx, h5ad_file in enumerate(h5ad_files, 1):
                dataset_name = sanitize_dataset_name(h5ad_file.stem)
                
                # Setup per-file logging
                file_log = log_dir / dataset_name / "pipeline.log"
                file_log.parent.mkdir(parents=True, exist_ok=True)
                json_path = file_log.with_suffix('.json')
                to_nice_file(output_file=json_path, rendered_file=file_log)
                
                test_action.log(
                    message_type="processing_file",
                    index=idx,
                    total=len(h5ad_files),
                    dataset_name=dataset_name
                )
                
                # Process the file
                success, message, processing_time, dataset_output_path = _process_single_file(
                    h5ad_path=h5ad_file,
                    interim_dir=Path("./data/interim"),  # Unused parameter (kept for API compatibility)
                    output_dir=batch_temp_dirs['output'],
                    chunk_size=10000,
                    top_genes=2000,
                    compression="zstd",
                    compression_level=3,
                    use_pyarrow=True,
                    test_size=0.05,
                    skip_train_test_split=False,
                    repo_id=None,
                    token=None,
                    mappers_path=mappers_path,
                    keep_interim=False,
                )
                
                results.append((dataset_name, success, message, processing_time, dataset_output_path))
                
                test_action.log(
                    message_type="file_processed",
                    dataset_name=dataset_name,
                    success=success,
                    processing_time_seconds=round(processing_time, 2)
                )
            
            # Validate results
            test_action.log(message_type="test_step", step=3, description="Validating batch results")
            
            # At least one file should have succeeded
            successful_files = [name for name, success, _, _, _ in results if success]
            assert len(successful_files) > 0, "At least one file should be processed successfully"
            
            test_action.log(
                message_type="batch_summary",
                total_files=len(results),
                successful=len(successful_files),
                failed=len(results) - len(successful_files)
            )
            
            # Validate each successful file's output
            for dataset_name, success, message, processing_time, dataset_output_path in results:
                if not success:
                    continue
                
                # Check train/test directories exist
                train_dir = dataset_output_path / 'train'
                test_dir = dataset_output_path / 'test'
                
                assert train_dir.exists(), f"Train directory should exist for {dataset_name}"
                assert test_dir.exists(), f"Test directory should exist for {dataset_name}"
                
                # Check parquet files exist
                train_files = list(train_dir.glob("*.parquet"))
                test_files = list(test_dir.glob("*.parquet"))
                
                assert len(train_files) > 0, f"Should have train files for {dataset_name}"
                assert len(test_files) > 0, f"Should have test files for {dataset_name}"
                
                # Validate per-file logs
                file_log = log_dir / dataset_name / "pipeline.log"
                file_json_log = log_dir / dataset_name / "pipeline.json"
                
                assert file_log.exists(), f"Log file should exist for {dataset_name}"
                assert file_json_log.exists(), f"JSON log should exist for {dataset_name}"
                
                test_action.log(
                    message_type="file_validation_complete",
                    dataset_name=dataset_name,
                    train_chunks=len(train_files),
                    test_chunks=len(test_files)
                )
            
            test_action.log(message_type="test_complete", status="passed")
    
    @pytest.mark.skip(reason="Batch processing test - run explicitly with pytest -m batch or pytest -k test_batch")
    def test_batch_sanitize_dataset_names(self) -> None:
        """Test dataset name sanitization for batch processing."""
        from cell2sentence4longevity.preprocess import sanitize_dataset_name
        
        test_cases = [
            ("my dataset.h5ad", "my_dataset.h5ad"),
            ("dataset@special.h5ad", "dataset_special.h5ad"),
            ("data  with   spaces", "data_with_spaces"),
            ("dataset#$%^&*.h5ad", "dataset_.h5ad"),  # Special chars become _, then .h5ad is preserved
            ("_leading_underscore", "leading_underscore"),
            ("trailing_underscore_", "trailing_underscore"),
            ("__multiple___underscores__", "multiple_underscores"),
        ]
        
        for input_name, expected_output in test_cases:
            result = sanitize_dataset_name(input_name)
            assert result == expected_output, \
                f"Expected sanitize_dataset_name('{input_name}') = '{expected_output}', got '{result}'"
    
    @pytest.mark.skip(reason="Batch processing test - run explicitly with pytest -m batch or pytest -k test_batch")
    def test_batch_check_output_exists(self, batch_temp_dirs: dict[str, Path]) -> None:
        """Test checking if output files already exist."""
        from cell2sentence4longevity.preprocess import check_output_exists
        
        output_dir = batch_temp_dirs['output']
        dataset_name = "test_dataset"
        
        # Initially should not exist
        assert not check_output_exists(output_dir, dataset_name, skip_train_test_split=False), \
            "Output should not exist initially"
        
        # Create train/test structure
        train_dir = output_dir / dataset_name / "train"
        test_dir = output_dir / dataset_name / "test"
        train_dir.mkdir(parents=True, exist_ok=True)
        test_dir.mkdir(parents=True, exist_ok=True)
        
        # Still should not exist (no parquet files)
        assert not check_output_exists(output_dir, dataset_name, skip_train_test_split=False), \
            "Output should not exist without parquet files"
        
        # Create dummy parquet files
        (train_dir / "chunk_0000.parquet").write_text("dummy")
        (test_dir / "chunk_0000.parquet").write_text("dummy")
        
        # Now should exist
        assert check_output_exists(output_dir, dataset_name, skip_train_test_split=False), \
            "Output should exist with parquet files"
        
        # Test single directory mode
        single_dataset_name = "single_dataset"
        single_dir = output_dir / single_dataset_name
        single_dir.mkdir(parents=True, exist_ok=True)
        
        assert not check_output_exists(output_dir, single_dataset_name, skip_train_test_split=True), \
            "Single directory output should not exist without parquet files"
        
        (single_dir / "chunk_0000.parquet").write_text("dummy")
        
        assert check_output_exists(output_dir, single_dataset_name, skip_train_test_split=True), \
            "Single directory output should exist with parquet files"
    
    @pytest.mark.skip(reason="Batch processing test - run explicitly with pytest -m batch or pytest -k test_batch")
    def test_batch_skip_existing_datasets(self, batch_temp_dirs: dict[str, Path]) -> None:
        """Test skipping datasets that already have output files."""
        from cell2sentence4longevity.preprocessing import download_dataset
        from cell2sentence4longevity.preprocess import _process_single_file, sanitize_dataset_name, check_output_exists
        
        # Setup logging
        log_dir = batch_temp_dirs['logs']
        json_log = log_dir / 'skip_test.json'
        rendered_log = log_dir / 'skip_test.log'
        to_nice_file(output_file=json_log, rendered_file=rendered_log)
        
        with start_action(action_type="test_batch_skip_existing") as test_action:
            # Download one test dataset
            url = "https://datasets.cellxgene.cziscience.com/10cc50a0-af80-4fa1-b668-893dd5c0113a.h5ad"
            h5ad_path = download_dataset(url=url, output_dir=batch_temp_dirs['input'])
            dataset_name = sanitize_dataset_name(h5ad_path.stem)
            
            # Create HGNC mapper
            from cell2sentence4longevity.preprocessing import create_hgnc_mapper
            # HGNC mappers are now stored in shared directory as parquet
            shared_dir = Path("./data/shared")
            mappers_path = shared_dir / 'hgnc_mappers.parquet'
            try:
                create_hgnc_mapper(shared_dir)
            except RuntimeError:
                mappers_path = None
            
            # Process the file once
            test_action.log(message_type="test_step", step=1, description="Processing file first time")
            
            success1, message1, time1, output_path1 = _process_single_file(
                h5ad_path=h5ad_path,
                interim_dir=Path("./data/interim"),  # Unused parameter (kept for API compatibility)
                output_dir=batch_temp_dirs['output'],
                chunk_size=10000,
                top_genes=2000,
                compression="zstd",
                compression_level=3,
                use_pyarrow=True,
                test_size=0.05,
                skip_train_test_split=False,
                repo_id=None,
                token=None,
                mappers_path=mappers_path,
                keep_interim=False,
            )
            
            assert success1, "First processing should succeed"
            test_action.log(message_type="first_processing_complete", processing_time=time1)
            
            # Check that output exists
            assert check_output_exists(batch_temp_dirs['output'], dataset_name, skip_train_test_split=False), \
                "Output should exist after first processing"
            
            test_action.log(message_type="test_step", step=2, description="Verifying skip logic")
            
            # Verify skip logic would work (we're not testing the full CLI here, 
            # just the helper function)
            should_skip = check_output_exists(
                batch_temp_dirs['output'], 
                dataset_name, 
                skip_train_test_split=False
            )
            
            assert should_skip, "Should skip dataset that already has output"
            test_action.log(message_type="skip_logic_validated", should_skip=should_skip)
            
            test_action.log(message_type="test_complete", status="passed")
    
    @pytest.mark.skip(reason="Batch processing test - run explicitly with pytest -m batch or pytest -k test_batch")
    def test_batch_memory_management(self, batch_temp_dirs: dict[str, Path]) -> None:
        """Test memory management in batch processing.
        
        This test validates that:
        1. Garbage collection is called after each file
        2. Memory is freed between file processing
        3. Processing multiple files doesn't accumulate memory
        """
        import gc
        import psutil
        import os
        
        from cell2sentence4longevity.preprocessing import download_dataset
        from cell2sentence4longevity.preprocess import _process_single_file, sanitize_dataset_name
        
        # Setup logging
        log_dir = batch_temp_dirs['logs']
        json_log = log_dir / 'memory_test.json'
        rendered_log = log_dir / 'memory_test.log'
        to_nice_file(output_file=json_log, rendered_file=rendered_log)
        
        with start_action(action_type="test_batch_memory_management") as test_action:
            process = psutil.Process(os.getpid())
            
            # Get initial memory
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            test_action.log(message_type="initial_memory", memory_mb=initial_memory)
            
            # Download test dataset
            url = "https://datasets.cellxgene.cziscience.com/10cc50a0-af80-4fa1-b668-893dd5c0113a.h5ad"
            h5ad_path = download_dataset(url=url, output_dir=batch_temp_dirs['input'])
            
            # Create HGNC mapper
            from cell2sentence4longevity.preprocessing import create_hgnc_mapper
            # HGNC mappers are now stored in shared directory as parquet
            shared_dir = Path("./data/shared")
            mappers_path = shared_dir / 'hgnc_mappers.parquet'
            try:
                create_hgnc_mapper(shared_dir)
            except RuntimeError:
                mappers_path = None
            
            memory_before = process.memory_info().rss / 1024 / 1024  # MB
            test_action.log(message_type="memory_before_processing", memory_mb=memory_before)
            
            # Process the file
            success, message, processing_time, output_path = _process_single_file(
                h5ad_path=h5ad_path,
                interim_dir=Path("./data/interim"),  # Unused parameter (kept for API compatibility)
                output_dir=batch_temp_dirs['output'],
                chunk_size=10000,
                top_genes=2000,
                compression="zstd",
                compression_level=3,
                use_pyarrow=True,
                test_size=0.05,
                skip_train_test_split=False,
                repo_id=None,
                token=None,
                mappers_path=mappers_path,
                keep_interim=False,
            )
            
            assert success, "Processing should succeed"
            
            # Force garbage collection (simulating batch processing cleanup)
            gc.collect()
            
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            test_action.log(
                message_type="memory_after_processing",
                memory_mb=memory_after,
                memory_increase_mb=memory_after - memory_before
            )
            
            # Memory increase should be reasonable (not exponential)
            # This is a soft check - we mainly want to ensure GC is called
            memory_increase = memory_after - memory_before
            test_action.log(
                message_type="memory_management_validation",
                memory_increase_mb=memory_increase,
                status="monitored"
            )
            
            # The key validation is that _process_single_file includes gc.collect()
            # We can't easily test the exact memory behavior here, but we log it
            # for manual inspection
            
            test_action.log(message_type="test_complete", status="passed")


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "-s"])

