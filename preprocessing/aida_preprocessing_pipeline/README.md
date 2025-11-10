# AIDA Dataset Preprocessing Pipeline

This pipeline transforms the AIDA (Asian Immune Diversity Atlas) h5ad file into a HuggingFace-ready dataset with cell sentences.

## Overview

**Input:** AIDA h5ad file (1.27M cells, 60K genes)
**Output:** HuggingFace dataset with train/test splits

The pipeline creates "cell sentences" - space-separated gene symbols ordered by expression level (top 2000 genes per cell).

## Dataset Information

- **Source:** CELLxGENE - AIDA India subset
- **File ID:** 9deda9ad-6a71-401e-b909-5263919d85f9
- **Download:** https://datasets.cellxgene.cziscience.com/9deda9ad-6a71-401e-b909-5263919d85f9.h5ad
- **License:** CC BY 4.0
- **Published Dataset:** https://huggingface.co/datasets/transhumanist-already-exists/aida-asian-pbmc-cell-sentence-top2000

## Pipeline Steps

### Step 1: Create HGNC Mapper
**Script:** `01_create_hgnc_mapper.py`

Downloads official gene mappings from HGNC to convert Ensembl IDs to gene symbols.

**Outputs:**
- `hgnc_mappers.pkl` - Bidirectional gene mappings

### Step 2: Convert H5AD to Parquet
**Script:** `02_convert_h5ad_to_parquet.py`

Converts h5ad expression matrix to cell sentences in parquet format.

**Process:**
1. Maps Ensembl IDs to gene symbols using HGNC
2. For each cell, extracts top 2000 expressed genes
3. Creates space-separated gene string (cell sentence)
4. Processes in chunks (10K cells) to manage memory
5. Saves to parquet files

**Outputs:**
- `temp_parquet/chunk_*.parquet` (127 files, ~12 GB)

### Step 3: Add Age and Cleanup
**Script:** `03_add_age_and_cleanup.py`

Adds age column and ensures proper column naming.

**Process:**
1. Extracts age from "development_stage" field (e.g., "22-year-old stage" → 22)
2. Renames columns if needed

**Outputs:**
- Updates all parquet files in place

### Step 4: Create Train/Test Split
**Script:** `04_create_train_test_split.py`

Creates stratified 95%/5% train/test split.

**Process:**
1. Loads all parquet chunks
2. Shuffles dataset (random_state=42)
3. Splits by age to maintain age distribution
4. Saves train and test sets separately

**Split Details:**
- **Train:** ~1,206,500 cells (95%)
- **Test:** ~63,500 cells (5%)
- **Stratification:** By age (21-31 years)
- **Random seed:** 42 (reproducible)

**Outputs:**
- `data_splits/train/chunk_*.parquet` (~121 files)
- `data_splits/test/chunk_*.parquet` (~7 files)

### Step 5: Upload to HuggingFace
**Script:** `05_upload_to_huggingface.py`

Uploads processed data to HuggingFace Hub with parallel uploads.

**Configuration Required:**
- HuggingFace token
- Repository name
- Username

**Outputs:**
- Public HuggingFace dataset

## Installation

### Requirements
- Python 3.10+
- 16+ GB RAM
- 30+ GB disk space

### Setup

```bash
# Install dependencies
pip install anndata pandas numpy h5py datasets huggingface-hub tqdm scikit-learn pyarrow

# Download AIDA h5ad file (place in parent directory)
wget https://datasets.cellxgene.cziscience.com/9deda9ad-6a71-401e-b909-5263919d85f9.h5ad
```

## Usage

Run scripts sequentially:

```bash
cd aida_preprocessing_pipeline

# Step 1: Create gene mappers
python 01_create_hgnc_mapper.py

# Step 2: Convert h5ad to parquet with cell sentences
python 02_convert_h5ad_to_parquet.py

# Step 3: Add age column and cleanup
python 03_add_age_and_cleanup.py

# Step 4: Create train/test split
python 04_create_train_test_split.py

# Step 5: Upload to HuggingFace (update config first!)
python 05_upload_to_huggingface.py
```

**Note:** Each script assumes the h5ad file is in the parent directory (`../9deda9ad-6a71-401e-b909-5263919d85f9.h5ad`).

## Expected Runtime

| Step | Time | Memory |
|------|------|--------|
| Step 1 | ~2 min | <1 GB |
| Step 2 | ~45 min | 8 GB |
| Step 3 | ~5 min | 4 GB |
| Step 4 | ~15 min | 16 GB |
| Step 5 | ~60 min | <1 GB |
| **Total** | **~2 hours** | **16 GB peak** |

## Output Structure

```
parent_directory/
├── 9deda9ad-6a71-401e-b909-5263919d85f9.h5ad  (input, 13 GB)
├── hgnc_mappers.pkl                            (from step 1)
├── temp_parquet/                               (from step 2)
│   ├── chunk_0000.parquet
│   └── ... (127 files, ~12 GB)
└── data_splits/                                (from step 4)
    ├── train/                                  (~11 GB)
    │   └── chunk_*.parquet (121 files)
    └── test/                                   (~0.6 GB)
        └── chunk_*.parquet (7 files)
```

## Final Dataset Schema

**Total Columns:** 50

**Key Fields:**
- `cell_sentence` (string): Top 2000 genes as space-separated string
- `age` (int): Donor age (21-31 years)
- `cell_type` (categorical): T cell, B cell, NK cell, etc.
- `sex` (categorical): male, female
- `donor_id` (categorical): 16 unique donors
- `tissue` (categorical): blood
- Plus 43 other metadata fields from original AIDA data

## Citation

When using this dataset, please cite:

1. **Original AIDA Project** - Asian Immune Diversity Atlas
2. **CELLxGENE** - Chan Zuckerberg Initiative Data Portal
3. **This Dataset** - https://huggingface.co/datasets/transhumanist-already-exists/aida-asian-pbmc-cell-sentence-top2000

## License

CC BY 4.0 (inherited from AIDA dataset)

## Support

For issues or questions:
- Open an issue on GitHub
- Check TRANSFORMATION_PIPELINE.md for detailed documentation
- Visit [CELLxGENE](https://cellxgene.cziscience.com/) for original data
