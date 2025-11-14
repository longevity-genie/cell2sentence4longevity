# Donor-Level Train/Test Split Implementation

## Overview

Modified the preprocessing pipeline to implement **donor-level train/test splitting** instead of cell-level splitting. This prevents data leakage of subject-specific signatures that could cause models to learn how to identify subjects rather than predict age.

## Problem Statement

Previously, the train/test split was performed at the **cell level** with age stratification. This meant that cells from the same donor could appear in both train and test sets. This creates **data leakage** because:

1. Each donor has unique biological signatures (genetics, environment, lifestyle, etc.)
2. The model can learn these donor-specific patterns
3. During evaluation, the model might recognize cells from the same donor seen during training
4. This leads to inflated performance metrics that don't reflect true generalization

## Solution

The new implementation performs **donor-level stratified splitting**:

1. **Group cells by donor** (using `donor_id`, `donor`, `subject_id`, or `individual` columns)
2. **Compute representative age** for each donor (median age of their cells)
3. **Split donors** (not cells) into train/test using stratified sampling by donor age
4. **Assign all cells** from each donor to the same split (train OR test, never both)

### Key Features

- **Donor-level stratification**: Maintains age distribution at the donor level
- **Automatic fallback**: Falls back to cell-level split if no donor column is found
- **Multiple donor column support**: Tries `donor_id`, `donor`, `subject_id`, `individual`
- **Comprehensive logging**: Logs donor statistics and split quality
- **Memory efficient**: Uses Polars lazy API and streaming where possible

## Files Modified

### 1. `src/cell2sentence4longevity/preprocessing/h5ad_converter.py`

**Function**: `convert_h5ad_to_train_test()`

**Changes**:
- Added donor column detection logic (lines 1342-1343)
- Implemented donor-level stratified split (lines 1383-1442)
- Computes median age per donor for stratification (lines 1395-1402)
- Uses sklearn stratified split on donors, then joins back to cells (lines 1412-1432)
- Falls back to cell-level split if no donor column found (lines 1345-1382)
- Added comprehensive logging for donor statistics

**Key Logic**:
```python
# Identify donor column
donor_cols = ['donor_id', 'donor', 'subject_id', 'individual']
donor_col = next((col for col in donor_cols if col in chunk_df.columns), None)

if donor_col:
    # Donor-level split
    donor_ages = chunk_df.group_by(donor_col).agg([
        pl.col('age').median().alias('donor_age')
    ])
    
    # Stratify donors by age
    train_donors = stratified_split_of_donors(...)
    test_donors = ...
    
    # Join back to get all cells for each donor
    train_chunk = chunk_df.join(train_donors, on=donor_col, how='inner')
    test_chunk = chunk_df.join(test_donors, on=donor_col, how='inner')
```

### 2. `src/cell2sentence4longevity/preprocessing/train_test_split.py`

**Function**: `create_train_test_split()`

**Changes**:
- Updated docstring to reflect donor-level splitting (lines 25-29)
- Added donor column detection for lazy datasets (lines 168-203)
- Implemented donor-level stratified split using sklearn (lines 212-272)
- Computes donor statistics (total donors, cells per donor) (lines 189-203)
- Uses sklearn's `train_test_split` with donor age stratification (lines 240-265)
- Filters lazy dataset using `is_in` with donor lists (lines 268-272)
- Falls back to cell-level split if no donor column (lines 274-287)

**Key Logic**:
```python
# Compute representative age for each donor
donor_ages = (
    lazy_dataset
    .group_by(donor_col)
    .agg([
        pl.col('age').median().alias('donor_age'),
        pl.len().alias('cell_count')
    ])
).collect()

# Stratified split of donors by age
train_donor_ids, test_donor_ids = sklearn_split(
    donor_ids_array,
    test_size=test_size,
    random_state=random_state,
    stratify=np.round(donor_ages_array, 1)
)

# Filter cells by donor assignment
lazy_train = lazy_dataset.filter(pl.col(donor_col).is_in(train_donor_series))
lazy_test = lazy_dataset.filter(pl.col(donor_col).is_in(test_donor_series))
```

### 3. `tests/test_donor_split.py` (New File)

**Purpose**: Comprehensive tests for donor-level split validation

**Tests**:
1. **`test_donor_split_no_leakage()`**: Verifies no donor appears in both train and test
   - Creates synthetic data with 10 donors, 50 cells each
   - Runs donor-level split
   - Validates no overlap between train and test donors
   - Verifies all cells from same donor are in same split

2. **`test_fallback_to_cell_level_split_without_donor_column()`**: Verifies fallback behavior
   - Creates data without donor column
   - Runs split
   - Validates cell-level split works correctly

## Benefits

1. **Prevents data leakage**: No donor-specific signatures can be learned and exploited
2. **True generalization**: Model must learn age-related patterns, not donor identification
3. **Better evaluation**: Test performance reflects true ability to predict age for unseen subjects
4. **Maintains age distribution**: Stratification ensures similar age distributions in train/test
5. **Robust fallback**: Automatically handles datasets without donor information
6. **Production ready**: Comprehensive logging and error handling

## Usage

No changes required to existing code! The donor-level split is **automatically applied** when:
- A donor column exists (`donor_id`, `donor`, `subject_id`, or `individual`)
- Age stratification is enabled (default: `stratify_by_age=True`)

If no donor column is found, it gracefully falls back to cell-level split with a warning.

### Example

```python
from cell2sentence4longevity.preprocessing import convert_h5ad_to_train_test

# Automatically uses donor-level split if donor column exists
convert_h5ad_to_train_test(
    h5ad_path=Path("data.h5ad"),
    output_dir=Path("output"),
    test_size=0.05,
    stratify_by_age=True,  # Stratifies at donor level!
    random_state=42
)
```

## Validation

Run the tests to verify donor-level split works correctly:

```bash
uv run pytest tests/test_donor_split.py -v
```

Expected output:
```
✓ Donor-level split validation passed:
  - Train donors: 8
  - Test donors: 2
  - Train cells: 400
  - Test cells: 100
  - No donor leakage detected!
```

## Logging

The implementation logs comprehensive donor statistics:

```json
{
  "message_type": "using_donor_level_split",
  "donor_col": "donor_id",
  "unique_donors_in_chunk": 47
}

{
  "message_type": "donor_statistics",
  "total_donors": 150,
  "total_cells": 45000,
  "avg_cells_per_donor": 300.0
}

{
  "message_type": "donor_split_complete",
  "train_donors": 142,
  "test_donors": 8
}
```

## Future Improvements

Potential enhancements (not implemented yet):

1. **Multiple stratification factors**: Stratify by both age and tissue type
2. **Minimum donor requirements**: Ensure test set has minimum number of donors per age bin
3. **Donor metadata export**: Export donor-level statistics for analysis
4. **Cross-validation support**: K-fold donor-level cross-validation

## References

- Original issue: "cells from the same donor do not appear in both train and test"
- Related concept: Group-aware splitting in machine learning
- Similar to: Patient-level splitting in medical ML, speaker-level splitting in audio ML

