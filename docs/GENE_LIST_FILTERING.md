# Gene List Filtering Feature

This document describes how to use the gene list filtering feature in the preprocessing pipeline.

## Overview

The gene list filtering feature allows you to create two types of gene sentences for each cell:
1. **full_gene_sentence**: Contains the top N (default 2000) genes sorted by expression
2. **cell_sentence**: Contains only genes from your gene lists, preserving the expression-based ordering

This is useful for focusing on specific gene sets (e.g., longevity-related genes, disease-associated genes) while maintaining the full gene expression context.

**By default**, the pipeline looks for gene lists in `./data/shared/gene_lists`. If this directory exists and contains `.txt` files, gene filtering is automatically enabled.

## Usage

### Default Behavior (Automatic)

If you have gene lists in the default location, just run the pipeline normally:

```bash
uv run preprocess run \
    --input-dir ./data/input \
    --output-dir ./data/output
```

The pipeline will automatically:
1. Check if `./data/shared/gene_lists` exists
2. Load all `.txt` files from that directory
3. Enable gene filtering if files are found
4. Proceed without filtering if the directory doesn't exist (no error)

### Custom Gene Lists Directory

To use a different directory:

```bash
uv run preprocess run \
    --gene-lists-dir /path/to/your/gene_lists \
    --input-dir ./data/input \
    --output-dir ./data/output
```

### Disabling Gene Filtering

To explicitly disable gene filtering (even if the default directory exists):

```bash
uv run preprocess run \
    --gene-lists-dir "" \
    --input-dir ./data/input \
    --output-dir ./data/output
```

Or provide a non-existent path:

```bash
uv run preprocess run \
    --gene-lists-dir /dev/null \
    --input-dir ./data/input \
    --output-dir ./data/output
```

### Gene List File Format

Gene list files should:
- Be in `.txt` format
- Contain one gene symbol per row
- Use standard gene symbols (e.g., "TP53", "BRCA1")

Example (`human_genage.txt`):
```
GHR
GHRH
SHC1
POU1F1
PROP1
TP53
TERC
TERT
```

### Directory Structure

Place all your gene list files in a single directory:

```
data/shared/gene_lists/
├── human_genage.txt          (308 genes)
├── opengenes.txt             (2001 genes)
└── clock_product_2k_top_genes.txt (2001 genes)
```

All genes from all files will be combined into a single gene set.

## Output

When gene list filtering is enabled, the output Parquet files will contain:

- **cell_sentence**: Space-separated list of genes from your gene lists, sorted by expression
- **full_gene_sentence**: Space-separated list of top 2K genes, sorted by expression
- All other standard columns (age, organism, tissue, etc.)

## Example

Given:
- Top 2K genes by expression: `GENE1 GENE2 TP53 GENE4 BRCA1 GENE6 ...`
- Gene list containing: `TP53, BRCA1, EGFR`
- Result:
  - `full_gene_sentence`: `GENE1 GENE2 TP53 GENE4 BRCA1 GENE6 ... (up to 2000 genes)`
  - `cell_sentence`: `TP53 BRCA1` (only genes in list, preserving order)

## Use Cases

1. **Longevity research**: Filter to known aging-related genes (GenAge, HAGR)
2. **Disease studies**: Focus on disease-specific gene panels
3. **Pathway analysis**: Analyze specific biological pathways
4. **Comparative studies**: Compare full vs. filtered gene signatures

## Performance

The filtering is implemented efficiently:
- Gene lists are loaded once at the start
- Filtering uses set membership (O(1) lookup)
- Expression ordering is preserved from the full list (no redundant sorting)

## Notes

- If the gene lists directory is not provided or empty, only `cell_sentence` is created (standard behavior)
- Ensembl IDs (starting with "ENS") are automatically filtered out
- Gene matching is case-sensitive
- Multiple gene list files are combined into a single filter set

