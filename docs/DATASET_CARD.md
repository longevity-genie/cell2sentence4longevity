---
license: cc-by-4.0
task_categories:
  - text-generation
  - token-classification
  - text-classification
language:
  - en
tags:
  - biology
  - single-cell
  - genomics
  - gene-expression
  - cell2sentence
  - age-prediction
  - longevity
size_categories:
  - 1M<n<10M
---

## Dataset Card: longevity-genie/cell2sentence4longevity-data

### Summary
This repository contains preprocessed single-cell RNA-seq (scRNA‑seq) datasets prepared as “cell sentences” for training and evaluation of cells2sentence-style models. Each cell is represented as a space‑separated sequence of top expressed gene symbols, enabling language‑model style training for tasks such as biological age prediction and other downstream applications.

This dataset targets fine‑tuning and evaluation of models inspired by cells2sentence approaches for cellular phenotyping, including age prediction as described in the preprint: [cells2sentence: Sequence models on gene expression](https://www.biorxiv.org/content/10.1101/2025.04.14.648850v3.full).

### What are “cell sentences”?
For each cell, we rank genes by expression and keep the top N (default 2000). We filter out Ensembl IDs and keep valid gene symbols, then serialize them as a whitespace‑separated string. This converts a numeric high‑dimensional cell profile into a token sequence amenable to language‑model training.

### Supported tasks and use cases
- Age prediction from single‑cell expression profiles
- Tissue/organ classification
- Cell type labeling and transfer
- Condition/disease stratification and dataset harmonization
- Few‑shot or instruction‑style fine‑tuning of sequence models on cells

### Data sources and provenance
- Source data are public scRNA‑seq h5ad datasets, primarily from the CZI CellxGene collections.
- When a dataset is detected as CellxGene (by UUID), we add `dataset_id` and, where available via cached collections metadata, join publication information:
  - `collection_id`, `publication_title`, `publication_doi`, `publication_description`, `publication_contact_name`, `publication_contact_email`.
- The pipeline is streaming and memory‑efficient, and uses Polars for processing.

### Repository structure
Each source dataset is organized under its own subfolder. There are two common layouts:
- Train/test split (default):
  - `<dataset_name>/train/chunk_*.parquet`
  - `<dataset_name>/test/chunk_*.parquet`
- Single split (if train/test split is disabled):
  - `<dataset_name>/chunk_*.parquet` or `<dataset_name>/chunks/chunk_*.parquet`

### Data fields (columns)
Columns are inherited from the input AnnData `.obs` table, plus generated fields:
- `cell_sentence` (string): space‑separated gene symbols for the cell (top‑N expression).
- `age` (float): numeric age extracted from `development_stage` where parsable (years). Cells with null age are filtered by default for training splits.
- `dataset_id` (string, optional): CellxGene dataset UUID when detected.
- Publication fields (optional, when join succeeds): `collection_id`, `publication_title`, `publication_doi`, `publication_description`, `publication_contact_name`, `publication_contact_email`.
- Other `.obs` fields (optional, dataset‑specific): e.g., `organism`, `tissue`, `cell_type`, `assay`, `sex`, `disease`, etc.

Notes:
- In current train/test outputs, the standardized column is `age` (years) when extractable from `development_stage`. Some upstream datasets encode mouse age in months; those may not map into `age` unless present in a parsable “year‑old” format.

### Preparation pipeline (high level)
1. Read h5ad in backed mode (streaming).
2. Map genes to symbols (HGNC lookup where helpful); filter out Ensembl IDs from sentences.
3. Build `cell_sentence` from top expressed genes per cell (default top‑N = 2000).
4. Extract `age` from `development_stage` when available (numeric years).
5. Optionally add `dataset_id` and join publication metadata if the dataset is found in CellxGene collections cache.
6. Filter cells with null `age` by default (for consistent age‑based tasks).
7. Write Parquet chunks and, by default, produce train/test split stratified by `age` (~95/5).

### How to use
Below is an example for downloading the repository snapshot and loading with Polars. This approach is scalable and keeps a local cache.

```python
from pathlib import Path
import polars as pl
from huggingface_hub import snapshot_download

repo_id = "longevity-genie/cell2sentence4longevity-data"
local_dir = Path(snapshot_download(repo_id=repo_id, repo_type="dataset"))

# Example: load train split for one dataset folder
dataset_name = "10cc50a0-af80-4fa1-b668-893dd5c0113a"  # replace with any available subfolder
train_glob = local_dir / dataset_name / "train" / "chunk_*.parquet"
test_glob = local_dir / dataset_name / "test" / "chunk_*.parquet"

train_df = pl.scan_parquet(str(train_glob)).collect()
test_df = pl.scan_parquet(str(test_glob)).collect()

# Basic checks
assert "cell_sentence" in train_df.columns
assert "age" in train_df.columns
```

You can iterate across all dataset subfolders to build training mixtures, or concatenate multiple datasets at scan‑time for large‑scale training pipelines.

### Limitations and caveats
- Not all datasets provide a reliably parsable human age; cells with null `age` are filtered for the default split.
- For mouse datasets that encode months (e.g., “24m”), month handling may appear in metadata extraction utilities but train/test outputs standardize on `age` when parsable as years.
- `.obs` schema varies across sources; presence of optional fields is dataset‑dependent.

### Licensing
- This repository aggregates preprocessed derivatives of public scRNA‑seq datasets. The original data remain under their respective licenses (see the source collection pages on CellxGene and corresponding publications). Please respect upstream licensing and citation requirements when using the data.
- The dataset card and pipeline code are provided under the project’s license; data licensing follows the upstream sources.

### Citation
If you use this dataset, please cite:
- cells2sentence preprint: “Sequence models on gene expression.” BioRxiv, 2025. [Link](https://www.biorxiv.org/content/10.1101/2025.04.14.648850v3.full)
- CellxGene data portal and the individual source publications for datasets included in this collection.

### Contact
Maintainer: `longevity-genie` on Hugging Face. Issues and improvements are welcome.
