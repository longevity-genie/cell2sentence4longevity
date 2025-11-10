#!/usr/bin/env python3
"""
Step 5: Upload train/test splits to HuggingFace hub.

This script uploads the processed data to HuggingFace with parallel uploads
for faster processing.

Inputs:
- data_splits/train/chunk_*.parquet (from Step 4)
- data_splits/test/chunk_*.parquet (from Step 4)
- README_AIDA.md (optional, for dataset card)

Outputs:
- Uploads to HuggingFace repository
"""

import os
from huggingface_hub import HfApi, login
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

print('='*80)
print('STEP 5: UPLOAD TO HUGGINGFACE')
print('='*80)

# Configuration - UPDATE THESE VALUES
HF_TOKEN = 'YOUR_HF_TOKEN_HERE'  # Replace with your HuggingFace token
DATASET_NAME = 'aida-asian-pbmc-cell-sentence-top2000'  # Repository name
USERNAME = 'transhumanist-already-exists'  # Your HuggingFace username

# Login to HuggingFace
print('\n🔐 Logging in to HuggingFace...')
login(token=HF_TOKEN)

# Setup
repo_id = f'{USERNAME}/{DATASET_NAME}'

# Create repository
api = HfApi()
print(f'\n🏗️  Creating/verifying repository: {repo_id}')
try:
    api.create_repo(
        repo_id=repo_id,
        repo_type='dataset',
        private=False,  # Set to True for private dataset
        exist_ok=True
    )
    print('   ✓ Repository ready')
except Exception as e:
    print(f'   Note: {e}')

# Check what's already uploaded
print(f'\n📋 Checking existing files...')
try:
    existing_files = set(api.list_repo_files(repo_id, repo_type='dataset'))
    print(f'   Found {len(existing_files)} files already uploaded')
except:
    existing_files = set()
    print('   No existing files found (new repository)')

# Upload README if exists
readme_path = '../README_AIDA.md'
if 'README.md' not in existing_files and os.path.exists(readme_path):
    print(f'\n📄 Uploading README...')
    api.upload_file(
        path_or_fileobj=readme_path,
        path_in_repo='README.md',
        repo_id=repo_id,
        repo_type='dataset',
        commit_message='Add README with dataset info'
    )
    print('   ✓ README uploaded')

def upload_file(file_info):
    """Upload a single file."""
    split, filename, file_path, repo_path = file_info
    try:
        api.upload_file(
            path_or_fileobj=file_path,
            path_in_repo=repo_path,
            repo_id=repo_id,
            repo_type='dataset',
            commit_message=None
        )
        return True, filename
    except Exception as e:
        return False, f'{filename}: {e}'

# Prepare train files
train_dir = '../data_splits/train'
train_files = sorted([f for f in os.listdir(train_dir) if f.endswith('.parquet')])
train_to_upload = []
for filename in train_files:
    repo_path = f'data/train/{filename}'
    if repo_path not in existing_files:
        file_path = f'{train_dir}/{filename}'
        train_to_upload.append(('train', filename, file_path, repo_path))

print(f'\n📤 Train files: {len(train_files)} total, {len(train_to_upload)} to upload')

# Prepare test files
test_dir = '../data_splits/test'
test_files = sorted([f for f in os.listdir(test_dir) if f.endswith('.parquet')])
test_to_upload = []
for filename in test_files:
    repo_path = f'data/test/{filename}'
    if repo_path not in existing_files:
        file_path = f'{test_dir}/{filename}'
        test_to_upload.append(('test', filename, file_path, repo_path))

print(f'📤 Test files: {len(test_files)} total, {len(test_to_upload)} to upload')

# Upload in parallel
all_to_upload = train_to_upload + test_to_upload
total = len(all_to_upload)

if total == 0:
    print('\n✅ All files already uploaded!')
else:
    print(f'\n🚀 Uploading {total} files in parallel (8 threads)...')

    max_workers = 8
    success_count = 0
    failed = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(upload_file, file_info): file_info for file_info in all_to_upload}

        with tqdm(total=total, desc='Uploading') as pbar:
            for future in as_completed(futures):
                success, result = future.result()
                if success:
                    success_count += 1
                else:
                    failed.append(result)
                pbar.update(1)

    print(f'\n✅ Upload complete!')
    print(f'   Successful: {success_count}/{total}')

    if failed:
        print(f'\n❌ Failed uploads ({len(failed)}):')
        for f in failed:
            print(f'   {f}')
    else:
        print(f'\n🎉 All files uploaded successfully!')
        print(f'   Dataset: https://huggingface.co/datasets/{repo_id}')
        print(f'   Train: {len(train_files)} files')
        print(f'   Test: {len(test_files)} files')

print(f'\n✅ PIPELINE COMPLETE!')
