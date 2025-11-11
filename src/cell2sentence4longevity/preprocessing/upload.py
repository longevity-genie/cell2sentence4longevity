"""HuggingFace upload module."""

from pathlib import Path
from typing import List, Tuple

import typer
from huggingface_hub import HfApi, login, CommitOperationAdd
from eliot import start_action
from tqdm import tqdm


DEFAULT_REPO_ID = "longevity-genie/cell2sentence4longevity-data"


def upload_to_huggingface(
    data_splits_dir: Path,
    token: str,
    repo_id: str = DEFAULT_REPO_ID,
    dataset_name: str | None = None,
    readme_path: Path | None = None
) -> bool:
    """Upload data to HuggingFace hub in a single commit.
    
    Handles two cases:
    1. Train/test splits: Uploads from train/test subdirectories
    2. Single dataset: Uploads from chunks directory without splitting
    
    Args:
        data_splits_dir: Directory containing train/test subdirectories, dataset_name/train/chunks/ structure,
                        or dataset_name/chunks/ structure (for non-split data)
        token: HuggingFace API token
        repo_id: HuggingFace repository ID (e.g., 'username/dataset-name'). 
                 Defaults to 'longevity-genie/cell2sentence4longevity-data'
        dataset_name: Name of the dataset. If None, inferred from directory structure
        readme_path: Optional path to README file
    """
    # Determine dataset name and directory structure
    # Check for different structure patterns
    has_train_test_split = False
    
    if dataset_name is None:
        # Try to infer from directory structure
        if (data_splits_dir / "train" / "chunks" / "chunk_0000.parquet").exists():
            # New structure with splits: data_splits_dir is output_dir, need to find dataset_name
            # Look for subdirectories that have train/chunks/
            has_train_test_split = True
            for subdir in data_splits_dir.iterdir():
                if subdir.is_dir() and (subdir / "train" / "chunks").exists():
                    dataset_name = subdir.name
                    break
        elif (data_splits_dir / "train" / "chunk_0000.parquet").exists():
            # Old structure with splits: data_splits_dir/train/ and data_splits_dir/test/
            # data_splits_dir is the dataset directory itself
            has_train_test_split = True
            dataset_name = data_splits_dir.name
        elif (data_splits_dir / "chunks" / "chunk_0000.parquet").exists():
            # New structure without splits: dataset_name/chunks/
            has_train_test_split = False
            dataset_name = data_splits_dir.name
        elif list(data_splits_dir.glob("chunk_*.parquet")):
            # Old flat structure without splits: chunks directly in directory
            has_train_test_split = False
            dataset_name = data_splits_dir.name
        else:
            # Assume it's the dataset directory itself
            dataset_name = data_splits_dir.name
            # Try to detect if it has splits
            has_train_test_split = (data_splits_dir / "train").exists()
    else:
        # Dataset name provided, detect if it has splits
        # When dataset_name is provided, data_splits_dir is typically already the dataset directory
        # Check if train/test directories exist directly in data_splits_dir
        has_train_test_split = (
            (data_splits_dir / "train" / "chunks").exists() or
            (data_splits_dir / "train").exists() or
            (data_splits_dir.parent / dataset_name / "train" / "chunks").exists() or
            (data_splits_dir.parent / dataset_name / "train").exists()
        )
    
    # Determine chunk directories based on split status
    if has_train_test_split:
        # Determine train and test chunk directories
        # When dataset_name is provided, data_splits_dir is typically the dataset directory itself
        # Check in order of most likely structure first
        
        # First check: files directly in train/ and test/ (most common case)
        if (data_splits_dir / "train" / "chunk_0000.parquet").exists() or list((data_splits_dir / "train").glob("chunk_*.parquet")):
            # Structure: data_splits_dir/train/ (data_splits_dir is dataset_name, files directly in train/)
            train_chunks_dir = data_splits_dir / "train"
            test_chunks_dir = data_splits_dir / "test"
        elif (data_splits_dir / "train" / "chunks" / "chunk_0000.parquet").exists() or list((data_splits_dir / "train" / "chunks").glob("chunk_*.parquet")) if (data_splits_dir / "train" / "chunks").exists() else False:
            # Structure: data_splits_dir/train/chunks/ (data_splits_dir is dataset_name)
            train_chunks_dir = data_splits_dir / "train" / "chunks"
            test_chunks_dir = data_splits_dir / "test" / "chunks"
        elif (data_splits_dir.parent / dataset_name / "train" / "chunk_0000.parquet").exists() or list((data_splits_dir.parent / dataset_name / "train").glob("chunk_*.parquet")) if (data_splits_dir.parent / dataset_name / "train").exists() else False:
            # Structure: data_splits_dir/dataset_name/train/ (data_splits_dir is output_dir, files directly in train/)
            train_chunks_dir = data_splits_dir.parent / dataset_name / "train"
            test_chunks_dir = data_splits_dir.parent / dataset_name / "test"
        elif (data_splits_dir.parent / dataset_name / "train" / "chunks" / "chunk_0000.parquet").exists() or list((data_splits_dir.parent / dataset_name / "train" / "chunks").glob("chunk_*.parquet")) if (data_splits_dir.parent / dataset_name / "train" / "chunks").exists() else False:
            # Structure: data_splits_dir/dataset_name/train/chunks/ (data_splits_dir is output_dir)
            train_chunks_dir = data_splits_dir.parent / dataset_name / "train" / "chunks"
            test_chunks_dir = data_splits_dir.parent / dataset_name / "test" / "chunks"
        else:
            # Fallback: assume chunks are directly in train/ and test/
            train_chunks_dir = data_splits_dir / "train"
            test_chunks_dir = data_splits_dir / "test"
        single_chunks_dir = None
    else:
        # No train/test split - single dataset
        if (data_splits_dir / "chunks" / "chunk_0000.parquet").exists():
            # New structure: dataset_name/chunks/
            single_chunks_dir = data_splits_dir / "chunks"
        elif (data_splits_dir / dataset_name / "chunks" / "chunk_0000.parquet").exists():
            # Structure: data_splits_dir/dataset_name/chunks/
            single_chunks_dir = data_splits_dir / dataset_name / "chunks"
        else:
            # Flat structure: chunks directly in directory
            single_chunks_dir = data_splits_dir
        train_chunks_dir = None
        test_chunks_dir = None
    
    # Validate dataset_name is not the hardcoded fallback
    if dataset_name == "dataset" and (data_splits_dir / "train").exists():
        # This should have been caught above, but if we still have "dataset", 
        # use the directory name as fallback
        dataset_name = data_splits_dir.name
    
    with start_action(
        action_type="upload_to_huggingface",
        repo_id=repo_id,
        data_splits_dir=str(data_splits_dir),
        dataset_name=dataset_name,
        has_train_test_split=has_train_test_split,
        train_chunks_dir=str(train_chunks_dir) if train_chunks_dir else None,
        test_chunks_dir=str(test_chunks_dir) if test_chunks_dir else None,
        single_chunks_dir=str(single_chunks_dir) if single_chunks_dir else None
    ) as action:
        # Login to HuggingFace
        action.log(message_type="logging_in")
        login(token=token)
        
        # Setup
        api = HfApi()
        
        # Create repository
        action.log(message_type="creating_repo")
        api.create_repo(
            repo_id=repo_id,
            repo_type='dataset',
            private=False,
            exist_ok=True
        )
        action.log(message_type="repo_ready")
        
        # Prepare operations list for batch commit
        operations: List[CommitOperationAdd] = []
        
        # Add README if provided
        if readme_path is not None and readme_path.exists():
            action.log(message_type="adding_readme_to_commit")
            operations.append(
                CommitOperationAdd(
                    path_in_repo='README.md',
                    path_or_fileobj=str(readme_path)
                )
            )
        
        if has_train_test_split:
            # Handle train/test split case
            # Prepare train files - upload all files regardless of existing status
            if train_chunks_dir is None or not train_chunks_dir.exists():
                action.log(
                    message_type="train_chunks_dir_not_found",
                    train_chunks_dir=str(train_chunks_dir) if train_chunks_dir else None,
                    data_splits_dir=str(data_splits_dir),
                    dataset_name=dataset_name,
                    has_train_test_split=has_train_test_split
                )
            train_files = sorted(list(train_chunks_dir.glob("chunk_*.parquet"))) if train_chunks_dir and train_chunks_dir.exists() else []
            action.log(
                message_type="train_files_search",
                train_chunks_dir=str(train_chunks_dir) if train_chunks_dir else None,
                train_chunks_dir_exists=train_chunks_dir.exists() if train_chunks_dir else False,
                files_found=len(train_files),
                sample_files=[str(f) for f in train_files[:3]]
            )
            for filepath in train_files:
                repo_path = f'{dataset_name}/train/{filepath.name}'
                operations.append(
                    CommitOperationAdd(
                        path_in_repo=repo_path,
                        path_or_fileobj=str(filepath)
                    )
                )
            
            action.log(
                message_type="train_files_prepared",
                total=len(train_files),
                to_upload=len(train_files),
                train_chunks_dir=str(train_chunks_dir)
            )
            
            # Prepare test files - upload all files regardless of existing status
            if test_chunks_dir is None or not test_chunks_dir.exists():
                action.log(
                    message_type="test_chunks_dir_not_found",
                    test_chunks_dir=str(test_chunks_dir) if test_chunks_dir else None,
                    data_splits_dir=str(data_splits_dir),
                    dataset_name=dataset_name
                )
            test_files = sorted(list(test_chunks_dir.glob("chunk_*.parquet"))) if test_chunks_dir and test_chunks_dir.exists() else []
            action.log(
                message_type="test_files_search",
                test_chunks_dir=str(test_chunks_dir) if test_chunks_dir else None,
                test_chunks_dir_exists=test_chunks_dir.exists() if test_chunks_dir else False,
                files_found=len(test_files),
                sample_files=[str(f) for f in test_files[:3]]
            )
            for filepath in test_files:
                repo_path = f'{dataset_name}/test/{filepath.name}'
                operations.append(
                    CommitOperationAdd(
                        path_in_repo=repo_path,
                        path_or_fileobj=str(filepath)
                    )
                )
            
            action.log(
                message_type="test_files_prepared",
                total=len(test_files),
                to_upload=len(test_files),
                test_chunks_dir=str(test_chunks_dir)
            )
            
            total_files = len(train_files) + len(test_files)
        else:
            # Handle single dataset case (no split) - upload all files regardless of existing status
            single_files = sorted(list(single_chunks_dir.glob("chunk_*.parquet"))) if single_chunks_dir.exists() else []
            for filepath in single_files:
                repo_path = f'{dataset_name}/{filepath.name}'
                operations.append(
                    CommitOperationAdd(
                        path_in_repo=repo_path,
                        path_or_fileobj=str(filepath)
                    )
                )
            
            action.log(
                message_type="single_dataset_files_prepared",
                total=len(single_files),
                to_upload=len(single_files),
                single_chunks_dir=str(single_chunks_dir)
            )
            
            total_files = len(single_files)
        
        # Upload all files in a single commit (always upload, even if files exist)
        if len(operations) == 0:
            action.log(
                message_type="no_files_to_upload", 
                dataset_name=dataset_name, 
                repo_id=repo_id,
                data_splits_dir=str(data_splits_dir),
                has_train_test_split=has_train_test_split,
                train_chunks_dir=str(train_chunks_dir) if train_chunks_dir else None,
                test_chunks_dir=str(test_chunks_dir) if test_chunks_dir else None
            )
            return False
        
        # Log the paths that will be used in the repo
        repo_paths = [op.path_in_repo for op in operations]
        action.log(
            message_type="starting_batch_upload", 
            total_operations=len(operations),
            total_files=total_files,
            repo_paths=repo_paths[:10]  # Log first 10 paths as sample
        )
        
        typer.echo(f"    Preparing to upload {total_files} files to {repo_id}...")
        
        # Create a single commit with all operations (will overwrite existing files)
        with tqdm(total=1, desc=f'Uploading {dataset_name}') as pbar:
            commit_info = api.create_commit(
                repo_id=repo_id,
                repo_type='dataset',
                operations=operations,
                commit_message=f'Upload {total_files} data files for {dataset_name}'
            )
            pbar.update(1)
        
        typer.echo(f"    Commit URL: {commit_info.commit_url}")
        
        action.log(
            message_type="upload_complete",
            commit_url=commit_info.commit_url,
            total_operations=len(operations),
            repo_id=repo_id,
            dataset_name=dataset_name
        )
        return True

