#!/usr/bin/env python3
"""
Step 3: Add age column and cleanup column names.

This script:
1. Extracts age as integer from development_stage field
2. Renames cell2sentence to cell_sentence (if needed)

Inputs:
- temp_parquet/chunk_*.parquet (from Step 2)

Outputs:
- Updates all parquet files in place
"""

import pandas as pd
import os
import re
from tqdm import tqdm

print('='*80)
print('STEP 3: ADD AGE COLUMN AND CLEANUP')
print('='*80)

parquet_dir = '../temp_parquet'
parquet_files = sorted([f for f in os.listdir(parquet_dir) if f.endswith('.parquet')])

print(f'\n📂 Found {len(parquet_files)} parquet files')

def extract_age(development_stage):
    """Extract age in years from development_stage string."""
    if pd.isna(development_stage):
        return None

    # Match patterns like "22-year-old stage"
    match = re.search(r'(\d+)-year-old', str(development_stage))
    if match:
        return int(match.group(1))
    return None

print('\n1️⃣  Adding age column...')
for filename in tqdm(parquet_files, desc='Adding age'):
    filepath = f'{parquet_dir}/{filename}'

    df = pd.read_parquet(filepath)

    # Add age column
    if 'age' not in df.columns:
        df['age'] = df['development_stage'].apply(extract_age)

    # Save back
    df.to_parquet(filepath)

print('   ✓ Age column added to all chunks')

print('\n2️⃣  Renaming columns if needed...')
sample_df = pd.read_parquet(f'{parquet_dir}/chunk_0000.parquet')

# Check if cell2sentence exists and needs renaming
if 'cell2sentence' in sample_df.columns:
    print('   Found cell2sentence column, renaming to cell_sentence...')
    for filename in tqdm(parquet_files, desc='Renaming'):
        filepath = f'{parquet_dir}/{filename}'

        df = pd.read_parquet(filepath)
        df = df.rename(columns={'cell2sentence': 'cell_sentence'})
        df.to_parquet(filepath)

    print('   ✓ Renamed cell2sentence → cell_sentence')
else:
    print('   ✓ Column naming already correct')

# Verify final structure
print('\n📊 Verification:')
df_test = pd.read_parquet(f'{parquet_dir}/chunk_0000.parquet')
print(f'   Total columns: {len(df_test.columns)}')
print(f'   age column present: {"age" in df_test.columns}')
print(f'   cell_sentence column present: {"cell_sentence" in df_test.columns}')

if 'age' in df_test.columns:
    print(f'\n   Sample age distribution:')
    age_counts = df_test['age'].value_counts().sort_index()
    for age, count in age_counts.items():
        print(f'     Age {age}: {count} cells')

print(f'\n✅ COMPLETE! Ready for Step 4.')
