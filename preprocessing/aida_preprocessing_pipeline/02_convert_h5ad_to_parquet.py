#!/usr/bin/env python3
"""
Step 2: Convert h5ad file to parquet with cell sentences.

This script transforms the AIDA h5ad file into parquet chunks, creating
"cell sentences" - space-separated gene symbols ordered by expression level.

Inputs:
- 9deda9ad-6a71-401e-b909-5263919d85f9.h5ad (must be in parent directory)
- hgnc_mappers.pkl (from Step 1)

Outputs:
- temp_parquet/chunk_*.parquet (127 files, ~10,000 cells each)
"""

import anndata as ad
import pandas as pd
import numpy as np
import pickle
import os
from tqdm import tqdm

print('='*80)
print('STEP 2: CONVERT H5AD TO PARQUET WITH CELL SENTENCES')
print('='*80)

# Load HGNC mappers
print('\n📥 Loading HGNC mappers...')
with open('../hgnc_mappers.pkl', 'rb') as f:
    mappers = pickle.load(f)

ensembl_to_symbol = mappers['ensembl_to_symbol']
print(f'   Loaded {len(ensembl_to_symbol):,} Ensembl → Symbol mappings')

# Load AIDA h5ad file
print('\n📥 Loading AIDA h5ad file...')
h5ad_path = '../9deda9ad-6a71-401e-b909-5263919d85f9.h5ad'
adata = ad.read_h5ad(h5ad_path, backed='r')

print(f'   Cells: {adata.n_obs:,}')
print(f'   Genes: {adata.n_vars:,}')

# Map Ensembl IDs to gene symbols
print('\n🔄 Mapping Ensembl IDs to gene symbols...')
ensembl_ids = list(adata.var_names)
gene_symbols = []

for i, ens_id in enumerate(ensembl_ids):
    if ens_id in ensembl_to_symbol:
        symbol = ensembl_to_symbol[ens_id]
        gene_symbols.append(symbol)
    else:
        # Fallback to feature_name if HGNC doesn't have mapping
        symbol = adata.var['feature_name'].iloc[i]
        gene_symbols.append(symbol)

print(f'   Mapped {len(gene_symbols):,} genes to symbols')

# Process in chunks
print('\n🧬 Creating cell sentences (top 2000 genes per cell)...')
print('   Processing in chunks to save memory...')

chunk_size = 10000
n_cells = adata.n_obs
n_chunks = (n_cells + chunk_size - 1) // chunk_size

# Create output directory
os.makedirs('../temp_parquet', exist_ok=True)

for chunk_idx in tqdm(range(n_chunks), desc='Processing chunks'):
    start_idx = chunk_idx * chunk_size
    end_idx = min(start_idx + chunk_size, n_cells)

    # Read chunk of expression matrix
    chunk_X = adata.X[start_idx:end_idx].toarray()

    # Get metadata for this chunk
    chunk_obs = adata.obs.iloc[start_idx:end_idx].copy()

    # Create cell sentences for each cell in chunk
    cell_sentences = []
    for cell_expr in chunk_X:
        # Get indices of top 2000 expressed genes
        top_gene_indices = np.argsort(cell_expr)[::-1][:2000]
        # Convert to gene symbols
        top_genes = [gene_symbols[idx] for idx in top_gene_indices]
        # Create space-separated string
        cell_sentence = ' '.join(top_genes)
        cell_sentences.append(cell_sentence)

    # Add cell_sentence column to metadata
    chunk_obs['cell_sentence'] = cell_sentences

    # Save chunk to parquet
    chunk_obs.to_parquet(f'../temp_parquet/chunk_{chunk_idx:04d}.parquet')

    # Clear memory
    del chunk_X, cell_sentences, chunk_obs

print(f'\n✅ Processed all {n_chunks} chunks')
print(f'   Output: ../temp_parquet/ ({n_chunks} files)')

adata.file.close()
print(f'\n✅ COMPLETE! Ready for Step 3.')
