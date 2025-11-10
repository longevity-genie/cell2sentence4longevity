#!/usr/bin/env bash
# Example usage script for cell2sentence preprocessing pipeline

# Set your HuggingFace token (or set as environment variable)
export HF_TOKEN="your_hf_token_here"

# Path to the AIDA h5ad file
H5AD_FILE="../9deda9ad-6a71-401e-b909-5263919d85f9.h5ad"

# Output directory
OUTPUT_DIR="./output"

# Repository ID for HuggingFace
REPO_ID="username/dataset-name"

echo "Example 1: Run complete pipeline"
echo "=================================="
uv run cell2sentence run-all "$H5AD_FILE" \
  --output-dir "$OUTPUT_DIR" \
  --repo-id "$REPO_ID" \
  --token "$HF_TOKEN" \
  --log-file ./logs/pipeline.log

echo ""
echo "Example 2: Run individual steps"
echo "=================================="

# Step 1: Create HGNC mapper
echo "Step 1: Creating HGNC mapper..."
uv run cell2sentence step1-hgnc-mapper \
  --output-dir "$OUTPUT_DIR" \
  --log-file ./logs/step1.log

# Step 2: Convert h5ad to parquet
echo "Step 2: Converting h5ad to parquet..."
uv run cell2sentence step2-convert-h5ad "$H5AD_FILE" \
  --mappers "$OUTPUT_DIR/hgnc_mappers.pkl" \
  --output-dir "$OUTPUT_DIR/temp_parquet" \
  --chunk-size 10000 \
  --top-genes 2000 \
  --log-file ./logs/step2.log

# Step 3: Add age and cleanup
echo "Step 3: Adding age and cleanup..."
uv run cell2sentence step3-add-age \
  --parquet-dir "$OUTPUT_DIR/temp_parquet" \
  --log-file ./logs/step3.log

# Step 4: Create train/test split
echo "Step 4: Creating train/test split..."
uv run cell2sentence step4-train-test-split \
  --parquet-dir "$OUTPUT_DIR/temp_parquet" \
  --output-dir "$OUTPUT_DIR/data_splits" \
  --test-size 0.05 \
  --random-state 42 \
  --log-file ./logs/step4.log

# Step 5: Upload to HuggingFace
echo "Step 5: Uploading to HuggingFace..."
uv run cell2sentence step5-upload \
  --data-splits-dir "$OUTPUT_DIR/data_splits" \
  --repo-id "$REPO_ID" \
  --token "$HF_TOKEN" \
  --max-workers 8 \
  --log-file ./logs/step5.log

echo ""
echo "Pipeline complete!"

