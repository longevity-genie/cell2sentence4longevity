#!/bin/bash
# Run complete AIDA preprocessing pipeline

set -e  # Exit on error

echo "========================================"
echo "AIDA PREPROCESSING PIPELINE"
echo "========================================"
echo ""

# Check if h5ad file exists
if [ ! -f "../9deda9ad-6a71-401e-b909-5263919d85f9.h5ad" ]; then
    echo "❌ Error: AIDA h5ad file not found in parent directory"
    echo "Please download it first:"
    echo "wget https://datasets.cellxgene.cziscience.com/9deda9ad-6a71-401e-b909-5263919d85f9.h5ad"
    exit 1
fi

echo "✓ Found AIDA h5ad file"
echo ""

# Step 1
echo "========================================"
echo "STEP 1: Create HGNC Mapper"
echo "========================================"
python 01_create_hgnc_mapper.py
if [ $? -ne 0 ]; then
    echo "❌ Step 1 failed"
    exit 1
fi
echo ""

# Step 2
echo "========================================"
echo "STEP 2: Convert H5AD to Parquet"
echo "========================================"
python 02_convert_h5ad_to_parquet.py
if [ $? -ne 0 ]; then
    echo "❌ Step 2 failed"
    exit 1
fi
echo ""

# Step 3
echo "========================================"
echo "STEP 3: Add Age and Cleanup"
echo "========================================"
python 03_add_age_and_cleanup.py
if [ $? -ne 0 ]; then
    echo "❌ Step 3 failed"
    exit 1
fi
echo ""

# Step 4
echo "========================================"
echo "STEP 4: Create Train/Test Split"
echo "========================================"
python 04_create_train_test_split.py
if [ $? -ne 0 ]; then
    echo "❌ Step 4 failed"
    exit 1
fi
echo ""

# Step 5 (optional - requires manual config)
echo "========================================"
echo "STEP 5: Upload to HuggingFace"
echo "========================================"
echo "⚠️  Please update 05_upload_to_huggingface.py with your HF token before running"
echo "Then run: python 05_upload_to_huggingface.py"
echo ""

echo "========================================"
echo "✅ PIPELINE COMPLETE (Steps 1-4)"
echo "========================================"
echo ""
echo "Outputs:"
echo "  - hgnc_mappers.pkl"
echo "  - temp_parquet/ (127 files)"
echo "  - data_splits/train/ (121 files)"
echo "  - data_splits/test/ (7 files)"
echo ""
echo "Next: Update and run 05_upload_to_huggingface.py"
