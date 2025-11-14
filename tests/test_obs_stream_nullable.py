from __future__ import annotations

from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import polars as pl
import pytest

from cell2sentence4longevity.preprocessing.obs_stream import (
    build_obs_chunk_dataframe,
    infer_obs_schema,
    list_obs_columns_from_group,
    preload_complex_obs_fields,
)


def test_nullable_obs_fields_stream_without_preload(tmp_path: Path) -> None:
    """Ensure nullable obs columns stream chunk-by-chunk without eager preloading."""
    original_setting = ad.settings.allow_write_nullable_strings
    ad.settings.allow_write_nullable_strings = True
    try:
        n_cells = 6
        obs = pd.DataFrame(
            {
                "nullable_int": pd.Series([1, None, 3, None, 5, 6], dtype="Int64"),
                "nullable_bool": pd.Series([True, None, False, True, None, False], dtype="boolean"),
            }
        )
        var = pd.DataFrame(index=[f"gene_{idx}" for idx in range(3)])
        adata = ad.AnnData(X=np.zeros((n_cells, var.shape[0])), obs=obs, var=var)
        h5ad_path = tmp_path / "nullable.h5ad"
        adata.write_h5ad(h5ad_path)
    finally:
        ad.settings.allow_write_nullable_strings = original_setting

    with h5py.File(h5ad_path, "r") as handle:
        obs_group = handle["obs"]
        fields = list_obs_columns_from_group(obs_group)
        schema = infer_obs_schema(obs_group)
        string_fields = {col for col, dtype in schema.items() if dtype == pl.String}
        preloaded = preload_complex_obs_fields(obs_group, fields, total_rows=n_cells)
        assert preloaded == {}

        chunk_df = build_obs_chunk_dataframe(
            obs_group=obs_group,
            fields=fields,
            start_idx=0,
            end_idx=n_cells,
            categorical_cache={},
            string_fields=string_fields,
            string_fill_value=None,
            preloaded_fields=preloaded,
        )
    assert chunk_df["nullable_int"].to_list() == [1, None, 3, None, 5, 6]
    assert chunk_df["nullable_bool"].to_list() == [True, None, False, True, None, False]

