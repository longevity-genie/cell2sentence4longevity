# File Size Limit Feature

## Overview

The batch processing functions in both `preprocess run` and `explore batch` now support a `--max-file-size-mb` option to skip files that exceed a specified size limit. This is particularly useful when processing large datasets with memory constraints.

## Usage Examples

### Limit to 12 GB (12,000 MB)

For batch preprocessing with the `preprocess run` command:

```bash
uv run preprocess run \
  --batch-mode \
  --input-dir ./data/input \
  --output-dir ./data/output \
  --max-file-size-mb 12000
```

For metadata extraction with the `explore batch` command:

```bash
uv run explore batch ./data/input \
  --output-dir ./data/output/meta \
  --max-file-size-mb 12000
```

### How It Works

1. **File Size Check**: Before processing each file, the command checks the file size in megabytes.

2. **Skip Large Files**: If the file size exceeds the specified limit, the file is skipped with a clear message:
   ```
   [1/5] Skipping dataset_name.h5ad (file size 15000.00 MB exceeds limit 12000.00 MB)
   ```

3. **Logging**: The skip event is logged with:
   - `message_type`: `skipping_large_file`
   - `file_size_mb`: Actual file size
   - `max_file_size_mb`: The configured limit

4. **Continue Processing**: Other files in the batch continue to be processed normally.

## Size Conversions

For reference, here are common size conversions to MB:

| Size | MB Value | Command Example |
|------|----------|-----------------|
| 1 GB | 1024 MB | `--max-file-size-mb 1024` |
| 5 GB | 5120 MB | `--max-file-size-mb 5120` |
| 10 GB | 10240 MB | `--max-file-size-mb 10240` |
| 12 GB | 12288 MB | `--max-file-size-mb 12288` |
| 50 GB | 51200 MB | `--max-file-size-mb 51200` |
| 100 GB | 102400 MB | `--max-file-size-mb 102400` |

Note: You can use approximate values like `12000` instead of the exact `12288` for simplicity.

## Complete Example

Process all h5ad files in a directory, but skip files larger than 12 GB:

```bash
# Preprocessing pipeline with file size limit
uv run preprocess run \
  --batch-mode \
  --input-dir ./data/input \
  --output-dir ./data/output \
  --max-file-size-mb 12000 \
  --skip-existing \
  --chunk-size 10000 \
  --compression zstd \
  --compression-level 3
```

```bash
# Metadata extraction with file size limit
uv run explore batch ./data/input \
  --output-dir ./data/output/meta \
  --max-file-size-mb 12000 \
  --skip-existing \
  --summary \
  --max-threads 4
```

## Integration with Other Features

The file size limit check works seamlessly with other batch processing features:

- **Skip Existing (`--skip-existing`)**: Files are checked for size before checking if output exists
- **Logging**: All skipped files are logged with detailed information
- **Batch Summaries**: Skipped files (due to size) appear in batch processing summaries with failure status
- **Error Handling**: File size checks happen before processing, preventing out-of-memory errors

## Benefits

1. **Memory Management**: Prevent out-of-memory crashes on systems with limited RAM
2. **Selective Processing**: Process only datasets that fit your resource constraints
3. **Batch Safety**: Continue processing other files even when some are too large
4. **Clear Feedback**: Immediate notification when files are skipped due to size
5. **Flexible Limits**: Set any size limit that matches your system's capabilities

