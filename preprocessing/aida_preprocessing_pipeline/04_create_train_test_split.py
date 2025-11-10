#!/usr/bin/env python3
"""
Step 4: Create stratified train/test split.

This script creates a 95%/5% train/test split stratified by age to maintain
age distribution in both sets.

Inputs:
- temp_parquet/chunk_*.parquet (from Step 3)

Outputs:
- data_splits/train/chunk_*.parquet (95% of data)
- data_splits/test/chunk_*.parquet (5% of data)
"""

import pandas as pd
import os
from sklearn.model_selection import train_test_split
from tqdm import tqdm

print('='*80)
print('STEP 4: CREATE STRATIFIED TRAIN/TEST SPLIT')
print('='*80)

# Load all parquet files
print('\n📥 Loading all parquet chunks...')
parquet_dir = '../temp_parquet'
parquet_files = sorted([f for f in os.listdir(parquet_dir) if f.endswith('.parquet')])

all_data = []
for filename in tqdm(parquet_files, desc='Loading'):
    filepath = f'{parquet_dir}/{filename}'
    df = pd.read_parquet(filepath)
    all_data.append(df)

# Concatenate all chunks
print('\n🔗 Concatenating all chunks...')
full_dataset = pd.concat(all_data, ignore_index=True)
print(f'   Total cells: {len(full_dataset):,}')

# Check age distribution
print('\n📊 Age distribution:')
age_counts = full_dataset['age'].value_counts().sort_index()
for age, count in age_counts.items():
    percentage = (count / len(full_dataset)) * 100
    print(f'   Age {age}: {count:,} cells ({percentage:.2f}%)')

# Shuffle dataset
print('\n🔀 Shuffling dataset...')
full_dataset = full_dataset.sample(frac=1, random_state=42).reset_index(drop=True)

# Stratified split by age
print('\n✂️  Creating stratified train/test split (95%/5% by age)...')
train_df, test_df = train_test_split(
    full_dataset,
    test_size=0.05,
    stratify=full_dataset['age'],
    random_state=42
)

print(f'\n✅ Split complete:')
print(f'   Train: {len(train_df):,} cells ({len(train_df)/len(full_dataset)*100:.1f}%)')
print(f'   Test:  {len(test_df):,} cells ({len(test_df)/len(full_dataset)*100:.1f}%)')

# Verify stratification
print('\n📊 Age distribution verification:')
print('\n   TRAIN SET:')
train_age_counts = train_df['age'].value_counts().sort_index()
for age, count in train_age_counts.items():
    percentage = (count / len(train_df)) * 100
    print(f'     Age {age}: {count:,} cells ({percentage:.2f}%)')

print('\n   TEST SET:')
test_age_counts = test_df['age'].value_counts().sort_index()
for age, count in test_age_counts.items():
    percentage = (count / len(test_df)) * 100
    print(f'     Age {age}: {count:,} cells ({percentage:.2f}%)')

# Save train and test sets
print('\n💾 Saving train and test sets...')

# Create output directories
os.makedirs('../data_splits/train', exist_ok=True)
os.makedirs('../data_splits/test', exist_ok=True)

# Save train set in chunks
chunk_size = 10000
n_train_chunks = (len(train_df) + chunk_size - 1) // chunk_size

print(f'\n   Saving train set ({n_train_chunks} chunks)...')
for i in tqdm(range(n_train_chunks), desc='Train chunks'):
    start_idx = i * chunk_size
    end_idx = min(start_idx + chunk_size, len(train_df))
    chunk = train_df.iloc[start_idx:end_idx]
    chunk.to_parquet(f'../data_splits/train/chunk_{i:04d}.parquet')

# Save test set in chunks
n_test_chunks = (len(test_df) + chunk_size - 1) // chunk_size

print(f'   Saving test set ({n_test_chunks} chunks)...')
for i in tqdm(range(n_test_chunks), desc='Test chunks'):
    start_idx = i * chunk_size
    end_idx = min(start_idx + chunk_size, len(test_df))
    chunk = test_df.iloc[start_idx:end_idx]
    chunk.to_parquet(f'../data_splits/test/chunk_{i:04d}.parquet')

print(f'\n✅ COMPLETE!')
print(f'\n   Train: {n_train_chunks} chunks in ../data_splits/train/')
print(f'   Test:  {n_test_chunks} chunks in ../data_splits/test/')

# Calculate sizes
train_size = sum(os.path.getsize(f'../data_splits/train/{f}')
                 for f in os.listdir('../data_splits/train'))
test_size = sum(os.path.getsize(f'../data_splits/test/{f}')
                for f in os.listdir('../data_splits/test'))

print(f'\n   Train size: {train_size / (1024**3):.2f} GB')
print(f'   Test size:  {test_size / (1024**3):.2f} GB')
print(f'\n✅ COMPLETE! Ready for Step 5 (upload).')
