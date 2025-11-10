# Comprehensive Logging Guide

This document describes the comprehensive logging system implemented in the cell2sentence4longevity preprocessing pipeline. The logging helps track data filtering, discarding, and potential issues when processing different datasets with varying metadata structures.

## Overview

The pipeline uses [Eliot](https://eliot.readthedocs.io/) for structured logging, which logs events in both JSON (machine-readable) and human-readable formats. All processing steps log detailed statistics about data filtering, discarding, and transformations.

## Logging Configuration

To enable logging to files, use the `--log-file` option with any CLI command:

```bash
preprocess step3_add_age --log-file logs/age_cleanup.log
```

This creates two files:
- `logs/age_cleanup.json` - Machine-readable structured logs
- `logs/age_cleanup.log` - Human-readable formatted logs

## Key Metrics Tracked

### 1. H5AD Loading (`load_h5ad_data`)

**What is logged:**
- Total number of cells and genes
- Available metadata columns in observations (`obs`) and variables (`var`)
- Presence of critical columns (`development_stage`, `feature_name`)
- Warnings for missing critical columns

**Log messages:**
- `h5ad_loaded` - Basic file statistics
- `metadata_structure` - Available columns
- `critical_columns_check` - Verification of required columns
- `warning_missing_column` - Alerts for missing critical columns

**Example:**
```json
{
  "message_type": "h5ad_loaded",
  "n_cells": 1000000,
  "n_genes": 20000
}
```

### 2. Gene Mapping (`map_genes_to_symbols`)

**What is logged:**
- Total genes to map
- Successfully mapped genes via HGNC
- Fallback mappings using feature_name
- Unmapped genes using Ensembl IDs directly
- Mapping percentages
- Sample unmapped genes (first 10) for debugging

**Log messages:**
- `starting_gene_mapping` - Initial count
- `gene_mapping_summary` - Detailed statistics with percentages
- `sample_unmapped_genes` - Examples of genes that couldn't be mapped

**Example:**
```json
{
  "message_type": "gene_mapping_summary",
  "total_genes": 20000,
  "mapped_via_hgnc": 18500,
  "mapped_via_fallback": 1200,
  "unmapped_using_ensembl_id": 300,
  "hgnc_mapping_percentage": 92.5,
  "fallback_percentage": 6.0,
  "unmapped_percentage": 1.5
}
```

### 3. Age Extraction and Cleanup (`add_age_and_cleanup`)

**What is logged:**
- Total cells before age extraction
- Cells with valid age values
- Cells with null/missing age values
- Age extraction percentages
- Sample development_stage values that failed to extract age
- Age distribution for valid ages

**Log messages:**
- `total_cells_before_age_extraction` - Initial cell count
- `age_extraction_summary` - Detailed statistics on success/failure
- `sample_null_age_development_stages` - Examples of failed extractions (first 10)
- `age_distribution` - Counts per age group

**Example:**
```json
{
  "message_type": "age_extraction_summary",
  "total_cells": 1000000,
  "cells_with_valid_age": 950000,
  "cells_with_null_age": 50000,
  "valid_age_percentage": 95.0,
  "null_age_percentage": 5.0
}
```

### 4. Train/Test Split (`create_train_test_split`)

**What is logged:**
- Total cells before filtering
- Cells with null ages detected
- Sample cells with null ages for debugging
- Cells filtered out due to null ages
- Ages with too few samples for stratification
- Train/test distribution statistics
- Final age distributions in train and test sets

**Log messages:**
- `total_cells_before_filtering` - Initial count
- `cells_with_null_age_detected` - Null age statistics
- `sample_cells_with_null_age` - Examples of problematic cells
- `filtering_summary` - Before/after filtering statistics
- `warning_ages_with_few_samples` - Ages that might cause stratification issues
- `split_complete` - Final train/test sizes
- `train_age_distribution` / `test_age_distribution` - Age distributions

**Example:**
```json
{
  "message_type": "filtering_summary",
  "cells_before_filtering": 1000000,
  "cells_after_filtering": 950000,
  "cells_discarded": 50000,
  "discard_percentage": 5.0
}
```

## Understanding Discard Reasons

### Age-Related Discarding

Cells can be discarded or have null ages for several reasons:

1. **Missing `development_stage` field**: The dataset doesn't have this metadata column
2. **Invalid format**: The development_stage value doesn't match the expected pattern (e.g., "22-year-old stage")
3. **Null values**: The development_stage is explicitly null or empty

**What to look for in logs:**
- `sample_null_age_development_stages` - Shows actual values that failed
- `null_age_percentage` - Indicates severity of the issue

### Gene Mapping Issues

Genes can fail to map for several reasons:

1. **Not in HGNC database**: Gene is too new or non-standard
2. **No `feature_name` fallback**: Dataset lacks this metadata column
3. **Ensembl ID not recognized**: Non-standard or deprecated ID

**What to look for in logs:**
- `sample_unmapped_genes` - Shows which genes couldn't be mapped
- `unmapped_percentage` - Indicates severity of the issue

### Stratification Warnings

Train/test split might fail if:

1. **Too few samples per age group**: Need at least `1/test_size` samples per age
2. **Highly imbalanced ages**: Some ages have very few cells

**What to look for in logs:**
- `warning_ages_with_few_samples` - Lists problematic age groups
- `problematic_ages` - Shows exact counts

## Example Use Cases

### Case 1: New Dataset with Different Metadata

```bash
preprocess run_all data/new_dataset.h5ad --log-file logs/new_dataset.log
```

**Check logs for:**
1. `metadata_structure` - Are the expected columns present?
2. `warning_missing_column` - What's missing?
3. `age_extraction_summary` - How many cells have valid ages?
4. `sample_null_age_development_stages` - What format is the age data in?

### Case 2: Debugging High Discard Rates

If you see high discard rates, examine:

1. **Age extraction issues:**
   ```bash
   grep "age_extraction_summary" logs/pipeline.log
   grep "sample_null_age_development_stages" logs/pipeline.log
   ```

2. **Gene mapping issues:**
   ```bash
   grep "gene_mapping_summary" logs/pipeline.log
   grep "sample_unmapped_genes" logs/pipeline.log
   ```

3. **Filter statistics:**
   ```bash
   grep "filtering_summary" logs/pipeline.log
   ```

### Case 3: Monitoring Production Pipeline

For production runs, monitor these key metrics:

- `null_age_percentage` < 10% (acceptable range)
- `hgnc_mapping_percentage` > 85% (good mapping)
- `discard_percentage` < 10% (low loss)
- No `warning_ages_with_few_samples` messages

## Log File Locations

By default, logs are saved to:
- `./logs/` directory when using `--log-file` option
- Integration tests save to `./logs/test_TIMESTAMP/`

## Programmatic Access

To access logs programmatically:

```python
import json
from pathlib import Path

# Read structured JSON logs
log_file = Path("logs/pipeline.json")
with open(log_file) as f:
    for line in f:
        event = json.loads(line)
        if event.get("message_type") == "age_extraction_summary":
            print(f"Valid ages: {event['valid_age_percentage']}%")
            print(f"Null ages: {event['null_age_percentage']}%")
```

## Troubleshooting

### Problem: High null age percentage

**Solution:**
1. Check `sample_null_age_development_stages` in logs
2. Verify the age format in your dataset
3. Consider modifying the `extract_age` regex pattern in `age_cleanup.py`

### Problem: Low HGNC mapping percentage

**Solution:**
1. Check `sample_unmapped_genes` in logs
2. Verify gene ID format (should be Ensembl IDs)
3. Check if dataset has `feature_name` column as fallback
4. Consider updating HGNC mapper data

### Problem: Stratification fails

**Solution:**
1. Check `warning_ages_with_few_samples` in logs
2. Consider reducing `test_size` to require fewer samples per age
3. Consider filtering out rare ages before splitting

## Best Practices

1. **Always use `--log-file`** when processing new datasets
2. **Review logs after each step** to catch issues early
3. **Archive logs** for reproducibility and debugging
4. **Monitor key percentages** to establish baseline expectations
5. **Share logs** when reporting issues or seeking help

## Summary Statistics

Each major step provides a summary with key metrics:

| Step | Key Metrics | Acceptable Ranges |
|------|-------------|-------------------|
| H5AD Loading | Column availability | All critical columns present |
| Gene Mapping | HGNC mapping % | > 85% |
| Age Extraction | Valid age % | > 90% |
| Train/Test Split | Discard % | < 10% |

These ranges are guidelines and may vary depending on your specific dataset and requirements.

