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
    add_age_and_cleanup,
    convert_h5ad_to_parquet,
    create_hgnc_mapper,
    create_train_test_split,
    download_dataset,
)


class TestIntegrationPipeline:
    """Integration tests for the full preprocessing pipeline."""
    
    @pytest.fixture
    def temp_dirs(self) -> dict[str, Path]:
        """Create test directories using project's data/ structure.
        
        Uses the standard data/input, data/interim, data/output pattern.
        Note: Input directory is shared across runs to avoid re-downloading.
              Interim/output directories are cleaned at the start but NOT at the end,
              allowing manual exploration.
        """
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Use project's data directories with test subdirectories
        project_root = Path(__file__).parent.parent
        data_dir = project_root / 'data'
        
        # Input is shared across test runs (don't delete downloads)
        # Interim/output/logs are timestamped for each run
        dirs = {
            'base': data_dir,
            'input': data_dir / 'input',  # Shared input directory for downloads
            'interim': data_dir / 'interim' / f'test_{timestamp}',
            'output': data_dir / 'output' / f'test_{timestamp}',
            'logs': project_root / 'logs' / f'test_{timestamp}',
        }
        
        # Clean up interim/output/logs for fresh start, but preserve input
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
        print(f"  - Interim: {dirs['interim']}")
        print(f"  - Output: {dirs['output']}")
        print(f"  - Logs: {dirs['logs']}")
        print(f"   (Files will be kept after test for manual exploration)")
        
        yield dirs
        
        # DO NOT cleanup - keep files for manual exploration
        print(f"\n✓ Test files preserved for manual exploration:")
        print(f"  - Input: {dirs['input']} (shared)")
        print(f"  - Interim: {dirs['interim']}")
        print(f"  - Output: {dirs['output']}")
        print(f"  - Logs: {dirs['logs']}")
    
    def test_full_pipeline_with_real_data(self, temp_dirs: dict[str, Path]) -> None:
        """Test the full pipeline with real data from CZI.
        
        This test:
        1. Downloads real data from CZI Science
        2. Runs the full preprocessing pipeline
        3. Validates that nothing crashes
        4. Checks logs are properly written
        5. Validates age field is numeric
        6. Checks output files exist and have correct structure
        """
        # Setup logging to files
        log_dir = temp_dirs['logs']
        json_log = log_dir / 'integration_test.json'
        rendered_log = log_dir / 'integration_test.log'
        
        to_nice_file(output_file=json_log, rendered_file=rendered_log)
        
        with start_action(action_type="integration_test_full_pipeline") as test_action:
            # Step 0: Download real dataset
            test_action.log(message_type="test_step", step=0, description="Downloading real dataset")
            url = "https://datasets.cellxgene.cziscience.com/9deda9ad-6a71-401e-b909-5263919d85f9.h5ad"
            
            h5ad_path = download_dataset(
                url=url,
                output_dir=temp_dirs['input'],
                filename="test_dataset.h5ad"
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
            mappers_path = temp_dirs['interim'] / 'hgnc_mappers.pkl'
            try:
                create_hgnc_mapper(temp_dirs['interim'])
                assert mappers_path.exists(), "HGNC mappers file should exist"
                test_action.log(message_type="hgnc_mapper_created", path=str(mappers_path))
            except RuntimeError as e:
                test_action.log(
                    message_type="hgnc_mapper_creation_failed", 
                    error=str(e),
                    fallback="Will use feature_name from h5ad"
                )
                mappers_path = None
            
            # Step 2: Convert h5ad to parquet
            test_action.log(message_type="test_step", step=2, description="Converting h5ad to parquet")
            parquet_dir = temp_dirs['interim'] / 'parquet_chunks'
            
            convert_h5ad_to_parquet(
                h5ad_path=h5ad_path,
                mappers_path=mappers_path,
                output_dir=parquet_dir,
                chunk_size=10000,
                top_genes=2000
            )
            
            # Validate parquet conversion
            parquet_files = list(parquet_dir.glob("chunk_*.parquet"))
            assert len(parquet_files) > 0, "Should have created parquet files"
            test_action.log(message_type="parquet_chunks_created", count=len(parquet_files))
            
            # Check first chunk structure
            sample_df = pl.read_parquet(parquet_files[0])
            test_action.log(
                message_type="sample_chunk_structure",
                columns=sample_df.columns,
                rows=len(sample_df)
            )
            assert 'cell_sentence' in sample_df.columns or 'cell2sentence' in sample_df.columns, \
                "Should have cell_sentence or cell2sentence column"
            
            # Step 3: Add age and cleanup
            test_action.log(message_type="test_step", step=3, description="Adding age and cleaning up")
            add_age_and_cleanup(parquet_dir)
            
            # Validate age column was added
            sample_df_after_age = pl.read_parquet(parquet_files[0])
            assert 'age' in sample_df_after_age.columns, "Should have age column"
            
            # Validate age field is numeric
            age_dtype = sample_df_after_age.schema['age']
            assert age_dtype in [pl.Int64, pl.Int32, pl.Int16, pl.Float64, pl.Float32], \
                f"Age column should be numeric, got {age_dtype}"
            test_action.log(message_type="age_column_validated", dtype=str(age_dtype))
            
            # Check age values are reasonable
            age_values = sample_df_after_age['age'].drop_nulls()
            if len(age_values) > 0:
                min_age = age_values.min()
                max_age = age_values.max()
                test_action.log(message_type="age_range", min_age=min_age, max_age=max_age)
                assert min_age >= 0, "Age should be non-negative"
                assert max_age <= 150, "Age should be realistic"
            
            # Validate column naming
            assert 'cell_sentence' in sample_df_after_age.columns, \
                "Should have cell_sentence column (renamed from cell2sentence if needed)"
            
            # Step 4: Create train/test split
            test_action.log(message_type="test_step", step=4, description="Creating train/test split")
            create_train_test_split(
                parquet_dir=parquet_dir,
                output_dir=temp_dirs['output'],
                test_size=0.05,
                random_state=42,
                chunk_size=10000
            )
            
            # Validate train/test split
            train_dir = temp_dirs['output'] / 'train'
            test_dir = temp_dirs['output'] / 'test'
            
            assert train_dir.exists(), "Train directory should exist"
            assert test_dir.exists(), "Test directory should exist"
            
            train_files = list(train_dir.glob("chunk_*.parquet"))
            test_files = list(test_dir.glob("chunk_*.parquet"))
            
            assert len(train_files) > 0, "Should have train files"
            assert len(test_files) > 0, "Should have test files"
            
            test_action.log(
                message_type="train_test_split_created",
                train_chunks=len(train_files),
                test_chunks=len(test_files)
            )
            
            # Verify train/test split ratio
            train_df = pl.scan_parquet(train_dir / "chunk_*.parquet")
            test_df = pl.scan_parquet(test_dir / "chunk_*.parquet")
            
            train_count = train_df.select(pl.count()).collect().item()
            test_count = test_df.select(pl.count()).collect().item()
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
            final_train_sample = pl.scan_parquet(train_dir / "chunk_*.parquet").limit(100).collect()
            
            assert 'cell_sentence' in final_train_sample.columns, "Final data should have cell_sentence"
            assert 'age' in final_train_sample.columns, "Final data should have age"
            
            # Check cell_sentence is not empty
            non_empty_sentences = final_train_sample['cell_sentence'].str.len_chars() > 0
            assert non_empty_sentences.sum() > 0, "Should have non-empty cell sentences"
            
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
            assert 'convert_h5ad_to_parquet' in log_content or 'h5ad' in log_content.lower(), \
                "Should log h5ad conversion"
            assert 'add_age_and_cleanup' in log_content or 'age' in log_content.lower(), \
                "Should log age processing"
            assert 'create_train_test_split' in log_content or 'split' in log_content.lower(), \
                "Should log train/test split"
            
            test_action.log(message_type="test_complete", status="passed")
    
    def test_pipeline_with_small_chunk_size(self, temp_dirs: dict[str, Path]) -> None:
        """Test pipeline with a smaller chunk size to verify chunking works correctly."""
        log_dir = temp_dirs['logs']
        json_log = log_dir / 'small_chunk_test.json'
        rendered_log = log_dir / 'small_chunk_test.log'
        
        to_nice_file(output_file=json_log, rendered_file=rendered_log)
        
        with start_action(action_type="integration_test_small_chunks") as test_action:
            # Download
            url = "https://datasets.cellxgene.cziscience.com/9deda9ad-6a71-401e-b909-5263919d85f9.h5ad"
            h5ad_path = download_dataset(
                url=url,
                output_dir=temp_dirs['input'],
                filename="test_dataset.h5ad"
            )
            
            # Create mapper (optional)
            mappers_path = temp_dirs['interim'] / 'hgnc_mappers.pkl'
            try:
                create_hgnc_mapper(temp_dirs['interim'])
            except RuntimeError:
                # HGNC download failed, will use feature_name fallback
                mappers_path = None
            
            # Convert with small chunk size
            parquet_dir = temp_dirs['interim'] / 'parquet_chunks'
            small_chunk_size = 5000
            
            convert_h5ad_to_parquet(
                h5ad_path=h5ad_path,
                mappers_path=mappers_path,
                output_dir=parquet_dir,
                chunk_size=small_chunk_size,
                top_genes=1000
            )
            
            # Verify chunks are approximately the right size
            parquet_files = list(parquet_dir.glob("chunk_*.parquet"))
            assert len(parquet_files) > 0
            
            # Check chunk sizes
            for pf in parquet_files[:3]:  # Check first 3 chunks
                df = pl.read_parquet(pf)
                # Last chunk might be smaller
                assert len(df) <= small_chunk_size * 1.1, \
                    f"Chunk should be around {small_chunk_size} rows, got {len(df)}"
            
            test_action.log(
                message_type="chunk_validation",
                num_chunks=len(parquet_files),
                chunk_size=small_chunk_size
            )
            
            # Continue with rest of pipeline
            add_age_and_cleanup(parquet_dir)
            
            create_train_test_split(
                parquet_dir=parquet_dir,
                output_dir=temp_dirs['output'],
                test_size=0.1,
                random_state=42,
                chunk_size=small_chunk_size
            )
            
            # Verify output
            train_files = list((temp_dirs['output'] / 'train').glob("chunk_*.parquet"))
            test_files = list((temp_dirs['output'] / 'test').glob("chunk_*.parquet"))
            
            assert len(train_files) > 0
            assert len(test_files) > 0
            
            test_action.log(
                message_type="small_chunk_test_complete",
                train_chunks=len(train_files),
                test_chunks=len(test_files)
            )


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
        from cell2sentence4longevity.preprocessing.age_cleanup import extract_age
        
        # Test various formats
        assert extract_age("22-year-old stage") == 22
        assert extract_age("45-year-old human stage") == 45
        assert extract_age("1-year-old stage") == 1
        assert extract_age("100-year-old stage") == 100
        
        # Test edge cases
        assert extract_age(None) is None
        assert extract_age("") is None
        assert extract_age("unknown") is None
        assert extract_age("child") is None
    
    def test_age_field_numeric_dtype(self) -> None:
        """Test that age field has numeric dtype after processing."""
        # Create a test dataframe
        df = pl.DataFrame({
            'development_stage': ['22-year-old stage', '45-year-old stage', '30-year-old stage'],
            'cell_sentence': ['GENE1 GENE2 GENE3', 'GENE4 GENE5 GENE6', 'GENE7 GENE8 GENE9']
        })
        
        # Apply age extraction
        from cell2sentence4longevity.preprocessing.age_cleanup import extract_age
        
        df = df.with_columns(
            pl.col('development_stage')
            .map_elements(extract_age, return_dtype=pl.Int64)
            .alias('age')
        )
        
        # Check dtype
        assert df.schema['age'] == pl.Int64, "Age should be Int64"
        
        # Check values
        assert df['age'].to_list() == [22, 45, 30], "Age values should be extracted correctly"
        
        # Verify all ages are numeric
        assert df['age'].dtype.is_integer(), "Age dtype should be integer"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "-s"])

