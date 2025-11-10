"""HuggingFace upload module."""

from pathlib import Path
from typing import List, Tuple

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
) -> None:
    """Upload train/test splits to HuggingFace hub in a single commit.
    
    Args:
        data_splits_dir: Directory containing train/test subdirectories or dataset_name/train/chunks/ structure
        token: HuggingFace API token
        repo_id: HuggingFace repository ID (e.g., 'username/dataset-name'). 
                 Defaults to 'longevity-genie/cell2sentence4longevity-data'
        dataset_name: Name of the dataset. If None, inferred from directory structure
        readme_path: Optional path to README file
    """
    # Determine dataset name and directory structure
    # Check for new structure: dataset_name/train/chunks/ and dataset_name/test/chunks/
    if dataset_name is None:
        # Try to infer from directory structure
        if (data_splits_dir / "train" / "chunks" / "chunk_0000.parquet").exists():
            # New structure: data_splits_dir is output_dir, need to find dataset_name
            # Look for subdirectories that have train/chunks/
            for subdir in data_splits_dir.iterdir():
                if subdir.is_dir() and (subdir / "train" / "chunks").exists():
                    dataset_name = subdir.name
                    break
        elif (data_splits_dir / "train" / "chunk_0000.parquet").exists():
            # Old structure: data_splits_dir/train/ and data_splits_dir/test/
            dataset_name = "dataset"
        else:
            # Assume it's the dataset directory itself
            dataset_name = data_splits_dir.name
    
    # Determine train and test chunk directories
    if (data_splits_dir / dataset_name / "train" / "chunks" / "chunk_0000.parquet").exists():
        # New structure: data_splits_dir/dataset_name/train/chunks/
        train_chunks_dir = data_splits_dir / dataset_name / "train" / "chunks"
        test_chunks_dir = data_splits_dir / dataset_name / "test" / "chunks"
    elif (data_splits_dir / "train" / "chunks" / "chunk_0000.parquet").exists():
        # New structure: data_splits_dir is dataset_name, train/chunks/ and test/chunks/
        train_chunks_dir = data_splits_dir / "train" / "chunks"
        test_chunks_dir = data_splits_dir / "test" / "chunks"
    elif (data_splits_dir / "train" / "chunk_0000.parquet").exists():
        # Old structure: data_splits_dir/train/ and data_splits_dir/test/
        train_chunks_dir = data_splits_dir / "train"
        test_chunks_dir = data_splits_dir / "test"
    else:
        # Fallback: assume chunks are directly in train/ and test/
        train_chunks_dir = data_splits_dir / "train"
        test_chunks_dir = data_splits_dir / "test"
    
    with start_action(
        action_type="upload_to_huggingface",
        repo_id=repo_id,
        data_splits_dir=str(data_splits_dir),
        dataset_name=dataset_name,
        train_chunks_dir=str(train_chunks_dir),
        test_chunks_dir=str(test_chunks_dir)
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
        
        # Check existing files
        action.log(message_type="checking_existing_files")
        existing_files = set(api.list_repo_files(repo_id, repo_type='dataset'))
        action.log(message_type="found_existing_files", count=len(existing_files))
        
        # Prepare operations list for batch commit
        operations: List[CommitOperationAdd] = []
        
        # Add README if provided and not exists
        if readme_path is not None and readme_path.exists() and 'README.md' not in existing_files:
            action.log(message_type="adding_readme_to_commit")
            operations.append(
                CommitOperationAdd(
                    path_in_repo='README.md',
                    path_or_fileobj=str(readme_path)
                )
            )
        
        # Prepare train files
        train_files = sorted(list(train_chunks_dir.glob("chunk_*.parquet"))) if train_chunks_dir.exists() else []
        train_to_upload = []
        for filepath in train_files:
            repo_path = f'data/{dataset_name}/train/chunks/{filepath.name}'
            if repo_path not in existing_files:
                train_to_upload.append(filepath)
                operations.append(
                    CommitOperationAdd(
                        path_in_repo=repo_path,
                        path_or_fileobj=str(filepath)
                    )
                )
        
        action.log(
            message_type="train_files_prepared",
            total=len(train_files),
            to_upload=len(train_to_upload),
            train_chunks_dir=str(train_chunks_dir)
        )
        
        # Prepare test files
        test_files = sorted(list(test_chunks_dir.glob("chunk_*.parquet"))) if test_chunks_dir.exists() else []
        test_to_upload = []
        for filepath in test_files:
            repo_path = f'data/{dataset_name}/test/chunks/{filepath.name}'
            if repo_path not in existing_files:
                test_to_upload.append(filepath)
                operations.append(
                    CommitOperationAdd(
                        path_in_repo=repo_path,
                        path_or_fileobj=str(filepath)
                    )
                )
        
        action.log(
            message_type="test_files_prepared",
            total=len(test_files),
            to_upload=len(test_to_upload),
            test_chunks_dir=str(test_chunks_dir)
        )
        
        # Upload all files in a single commit
        total_files = len(train_to_upload) + len(test_to_upload)
        if len(operations) == 0:
            action.log(message_type="no_files_to_upload")
        else:
            action.log(
                message_type="starting_batch_upload", 
                total_operations=len(operations),
                total_files=total_files
            )
            
            # Create a single commit with all operations
            with tqdm(total=1, desc='Uploading batch') as pbar:
                commit_info = api.create_commit(
                    repo_id=repo_id,
                    repo_type='dataset',
                    operations=operations,
                    commit_message=f'Upload {total_files} data files'
                )
                pbar.update(1)
            
            action.log(
                message_type="upload_complete",
                commit_url=commit_info.commit_url,
                total_operations=len(operations)
            )

