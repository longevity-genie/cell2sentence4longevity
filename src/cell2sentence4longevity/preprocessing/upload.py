"""HuggingFace upload module."""

from pathlib import Path
from typing import List, Tuple

from huggingface_hub import HfApi, login, CommitOperationAdd
from eliot import start_action
from tqdm import tqdm


def upload_to_huggingface(
    data_splits_dir: Path,
    repo_id: str,
    token: str,
    readme_path: Path | None = None,
    max_workers: int = 8
) -> None:
    """Upload train/test splits to HuggingFace hub in a single commit.
    
    Args:
        data_splits_dir: Directory containing train/test subdirectories
        repo_id: HuggingFace repository ID (e.g., 'username/dataset-name')
        token: HuggingFace API token
        readme_path: Optional path to README file
        max_workers: Number of parallel upload threads (unused, kept for backwards compatibility)
    """
    with start_action(
        action_type="upload_to_huggingface",
        repo_id=repo_id,
        data_splits_dir=str(data_splits_dir)
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
        train_dir = data_splits_dir / "train"
        train_files = sorted(list(train_dir.glob("chunk_*.parquet")))
        train_to_upload = []
        for filepath in train_files:
            repo_path = f'data/train/{filepath.name}'
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
            to_upload=len(train_to_upload)
        )
        
        # Prepare test files
        test_dir = data_splits_dir / "test"
        test_files = sorted(list(test_dir.glob("chunk_*.parquet")))
        test_to_upload = []
        for filepath in test_files:
            repo_path = f'data/test/{filepath.name}'
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
            to_upload=len(test_to_upload)
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

