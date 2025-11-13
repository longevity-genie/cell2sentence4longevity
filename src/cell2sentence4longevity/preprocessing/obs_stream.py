"""Utilities for streaming AnnData.obs directly from HDF5 without pandas."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Sequence, Set

import h5py
import numpy as np
import polars as pl

from anndata._io.specs import read_elem


def _decode_scalar(value: Any) -> Any:
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        return value.item()
    return value


def _load_categorical_values(column_group: h5py.Group) -> np.ndarray:
    raw_values = column_group["categories"][()]
    if raw_values.size == 0:
        return np.empty(0, dtype=object)
    converted = np.empty(len(raw_values), dtype=object)
    for idx, value in enumerate(raw_values):
        converted[idx] = _decode_scalar(value)
    return converted


def _decode_categorical_codes(
    column_group: h5py.Group,
    categories: np.ndarray,
    start_idx: int,
    end_idx: int
) -> np.ndarray:
    codes = column_group["codes"][start_idx:end_idx]
    if codes.size == 0:
        return np.empty(0, dtype=object)
    codes_array = np.asarray(codes, dtype=np.int64)
    result = np.empty(len(codes_array), dtype=object)
    result.fill(None)
    if categories.size == 0:
        return result
    mask = codes_array >= 0
    if np.any(mask):
        valid_codes = codes_array[mask].astype(np.intp, copy=False)
        decoded = np.asarray(categories[valid_codes], dtype=object)
        result[mask] = decoded
    return result


def _read_dataset_slice(
    dataset: h5py.Dataset,
    start_idx: int,
    end_idx: int
) -> Any:
    data = dataset[start_idx:end_idx]
    if not hasattr(data, "dtype"):
        return data
    dtype_kind = data.dtype.kind
    if dtype_kind == "S":
        return [_decode_scalar(value) for value in data]
    if dtype_kind == "U":
        return data.tolist()
    if dtype_kind == "O":
        return [_decode_scalar(value) for value in data]
    return data


def _read_obs_field_slice(
    field_node: h5py.Dataset | h5py.Group,
    field_name: str,
    start_idx: int,
    end_idx: int,
    categorical_cache: Dict[str, np.ndarray]
) -> Any:
    encoding_type = field_node.attrs.get("encoding-type", "array")
    if encoding_type == "categorical":
        if not isinstance(field_node, h5py.Group):
            msg = f"Unexpected categorical node type for field '{field_name}'"
            raise ValueError(msg)
        if field_name not in categorical_cache:
            categorical_cache[field_name] = _load_categorical_values(field_node)
        categories = categorical_cache[field_name]
        return _decode_categorical_codes(field_node, categories, start_idx, end_idx)
    if not isinstance(field_node, h5py.Dataset):
        msg = f"Unsupported obs field storage for '{field_name}'"
        raise ValueError(msg)
    return _read_dataset_slice(field_node, start_idx, end_idx)


def _ensure_numpy_array(values: Any) -> np.ndarray:
    if isinstance(values, np.ndarray):
        return values
    return np.asarray(values)


def _fill_string_array(arr: np.ndarray, fill_value: str) -> np.ndarray:
    arr = np.asarray(arr, dtype=object)
    mask = np.equal(arr, None)
    if mask.any():
        arr = arr.copy()
        arr[mask] = fill_value
    return arr


def _to_string_list(values: np.ndarray) -> list[Any]:
    result: list[Any] = []
    for value in values.tolist():
        if value is None:
            result.append(None)
        else:
            result.append(str(value))
    return result


def _normalize_preloaded_column(values: Any, total_rows: int) -> np.ndarray:
    if isinstance(values, np.ndarray):
        if values.ndim == 0:
            filled = np.full(total_rows, _decode_scalar(values.item()), dtype=object)
            return filled
        return values
    if hasattr(values, "to_numpy"):
        arr = values.to_numpy()
        arr = np.asarray(arr, dtype=object)
        if arr.ndim == 0:
            arr = np.full(total_rows, _decode_scalar(arr.item()), dtype=object)
        return arr
    if hasattr(values, "to_list"):
        arr = np.asarray(values.to_list(), dtype=object)
    elif hasattr(values, "tolist"):
        arr = np.asarray(values.tolist(), dtype=object)
    else:
        try:
            arr = np.asarray(values, dtype=object)
        except Exception:
            arr = np.array([values], dtype=object)
    if arr.ndim == 0:
        arr = np.full(total_rows, _decode_scalar(arr.item()), dtype=object)
    return arr


def _slice_preloaded_column(
    data: np.ndarray | Sequence[Any],
    start_idx: int,
    end_idx: int
) -> Any:
    if isinstance(data, np.ndarray):
        return data[start_idx:end_idx]
    return data[start_idx:end_idx]


def read_obs_chunk_dict(
    obs_group: h5py.Group,
    fields: list[str],
    start_idx: int,
    end_idx: int,
    categorical_cache: Dict[str, np.ndarray] | None = None,
    as_lists: bool = False,
    string_fields: Set[str] | None = None,
    string_fill_value: str | None = None,
    preloaded_fields: Dict[str, np.ndarray | Sequence[Any]] | None = None,
) -> dict[str, Any]:
    if categorical_cache is None:
        categorical_cache = {}
    if preloaded_fields is None:
        preloaded_fields = {}
    chunk_data: dict[str, Any] = {}
    for field in fields:
        if field in preloaded_fields:
            values = _slice_preloaded_column(
                preloaded_fields[field],
                start_idx,
                end_idx
            )
        else:
            node = obs_group.get(field)
            if node is None:
                msg = f"Field '{field}' not found in obs group"
                raise KeyError(msg)
            values = _read_obs_field_slice(
                node,
                field,
                start_idx,
                end_idx,
                categorical_cache
            )
        is_string_field = string_fields is not None and field in string_fields
        if not as_lists or is_string_field:
            values = _ensure_numpy_array(values)
        if is_string_field:
            if string_fill_value is not None:
                values = _fill_string_array(values, string_fill_value)
            values_list = _to_string_list(values)
            values = pl.Series(field, values_list, dtype=pl.String, strict=False)
        if as_lists:
            if isinstance(values, np.ndarray):
                values = values.tolist()
            chunk_data[field] = values
        else:
            chunk_data[field] = values
    return chunk_data


def build_obs_chunk_dataframe(
    obs_group: h5py.Group,
    fields: list[str],
    start_idx: int,
    end_idx: int,
    categorical_cache: Dict[str, np.ndarray] | None = None,
    string_fields: Set[str] | None = None,
    string_fill_value: str | None = None,
    preloaded_fields: Dict[str, np.ndarray | Sequence[Any]] | None = None,
) -> pl.DataFrame:
    chunk_dict = read_obs_chunk_dict(
        obs_group=obs_group,
        fields=fields,
        start_idx=start_idx,
        end_idx=end_idx,
        categorical_cache=categorical_cache,
        as_lists=False,
        string_fields=string_fields,
        string_fill_value=string_fill_value,
        preloaded_fields=preloaded_fields
    )
    return pl.DataFrame(chunk_dict)


def preload_complex_obs_fields(
    obs_group: h5py.Group,
    fields: list[str],
    total_rows: int
) -> dict[str, np.ndarray]:
    preloaded: dict[str, np.ndarray] = {}
    for field in fields:
        node = obs_group.get(field)
        if node is None:
            continue
        encoding_type = node.attrs.get("encoding-type", "array")
        is_dataset = isinstance(node, h5py.Dataset)
        if is_dataset or encoding_type == "categorical":
            continue
        try:
            values = read_elem(node)
        except Exception as exc:
            msg = f"Failed to preload obs field '{field}': {exc}"
            raise RuntimeError(msg) from exc
        preloaded[field] = _normalize_preloaded_column(values, total_rows)
    return preloaded


def list_obs_columns_from_group(obs_group: h5py.Group) -> list[str]:
    return [name for name in obs_group.keys() if name != "index"]


def list_obs_columns_from_file(h5ad_path: Path) -> list[str]:
    with h5py.File(h5ad_path, "r") as handle:
        obs_group = handle["obs"]
        return list_obs_columns_from_group(obs_group)


def infer_obs_schema(obs_group: h5py.Group) -> dict[str, pl.datatypes.DataType]:
    """Infer Polars dtypes for each obs column based on HDF5 metadata."""
    schema: dict[str, pl.datatypes.DataType] = {}
    for name, node in obs_group.items():
        if name == "index":
            continue
        encoding_type = node.attrs.get("encoding-type", "array")
        if encoding_type in {"categorical", "string-array"}:
            schema[name] = pl.String
            continue
        if encoding_type == "nullable-boolean":
            schema[name] = pl.Boolean
            continue
        if encoding_type == "nullable-integer":
            schema[name] = pl.Int64
            continue
        if encoding_type == "nullable-string-array":
            schema[name] = pl.String
            continue
        if not isinstance(node, h5py.Dataset):
            continue
        kind = node.dtype.kind
        if kind == "b":
            schema[name] = pl.Boolean
        elif kind in {"i", "u"}:
            schema[name] = pl.Int64
        elif kind == "f":
            schema[name] = pl.Float64
        elif kind in {"S", "U", "O"}:
            schema[name] = pl.String
        else:
            schema[name] = pl.String
    return schema

