from __future__ import annotations

from pathlib import Path
from typing import Optional, Callable, Dict, Any
import json
import time
import math

import numpy as np
import typer
from eliot import start_action, to_file
from pycomfort.logging import to_nice_file, to_nice_stdout

from cell2sentence4longevity.preprocessing.h5ad_converter import create_cell_sentence


app = typer.Typer(
    name="bench-h5ad",
    help="Benchmark variants of create_cell_sentence for efficiency",
    add_completion=False,
)


def variant_np_isin(
    cell_expr: np.ndarray,
    gene_symbols: np.ndarray,
    top_n: Optional[int] = 2000,
    gene_lists: Optional[dict[str, set[str]]] = None,
) -> dict[str, str]:
    # Filter out Ensembl IDs using vectorized operations
    if gene_symbols.dtype.kind in ("U", "S", "O"):
        gene_symbols_str = (
            gene_symbols.astype(str) if gene_symbols.dtype == object else gene_symbols
        )
        valid_mask = ~np.char.startswith(gene_symbols_str, "ENS")
    else:
        valid_mask = np.array(
            [not str(gene).startswith("ENS") for gene in gene_symbols], dtype=bool
        )

    valid_indices = np.where(valid_mask)[0]
    if len(valid_indices) == 0:
        result = {"gene_sentence_all": "", f"gene_sentence_{top_n}": ""}
        if gene_lists:
            for filestem in gene_lists:
                result[f"gene_sentence_{filestem}"] = ""
        return result

    valid_expr = cell_expr[valid_indices]
    valid_symbols = gene_symbols[valid_indices]

    sorted_indices = np.argsort(valid_expr)[::-1]
    all_genes_sorted_array = valid_symbols[sorted_indices]

    result: dict[str, str] = {}

    # Join directly from numpy array (avoid tolist) to compare with baseline
    result["gene_sentence_all"] = " ".join(all_genes_sorted_array)

    if top_n is None:
        pass
    else:
        if top_n < len(all_genes_sorted_array):
            result[f"gene_sentence_{top_n}"] = " ".join(all_genes_sorted_array[:top_n])
        else:
            result[f"gene_sentence_{top_n}"] = result["gene_sentence_all"]

    if gene_lists:
        for filestem, gene_set in gene_lists.items():
            # Use numpy.isin for vectorized membership, preserves order
            # Convert set to array once
            if len(gene_set) == 0:
                result[f"gene_sentence_{filestem}"] = ""
                continue
            list_arr = np.fromiter(gene_set, dtype=object, count=len(gene_set))
            mask = np.isin(all_genes_sorted_array, list_arr, assume_unique=False)
            filtered = all_genes_sorted_array[mask]
            result[f"gene_sentence_{filestem}"] = " ".join(filtered)

    return result


def variant_join_no_tolist(
    cell_expr: np.ndarray,
    gene_symbols: np.ndarray,
    top_n: Optional[int] = 2000,
    gene_lists: Optional[dict[str, set[str]]] = None,
) -> dict[str, str]:
    # Same as baseline except avoid tolist() when joining
    if gene_symbols.dtype.kind in ("U", "S", "O"):
        gene_symbols_str = (
            gene_symbols.astype(str) if gene_symbols.dtype == object else gene_symbols
        )
        valid_mask = ~np.char.startswith(gene_symbols_str, "ENS")
    else:
        valid_mask = np.array(
            [not str(gene).startswith("ENS") for gene in gene_symbols], dtype=bool
        )

    valid_indices = np.where(valid_mask)[0]
    if len(valid_indices) == 0:
        result = {"gene_sentence_all": "", f"gene_sentence_{top_n}": ""}
        if gene_lists:
            for filestem in gene_lists:
                result[f"gene_sentence_{filestem}"] = ""
        return result

    valid_expr = cell_expr[valid_indices]
    valid_symbols = gene_symbols[valid_indices]
    sorted_indices = np.argsort(valid_expr)[::-1]
    all_genes_sorted_array = valid_symbols[sorted_indices]

    result: dict[str, str] = {}
    result["gene_sentence_all"] = " ".join(all_genes_sorted_array)
    if top_n is None:
        pass
    else:
        if top_n < len(all_genes_sorted_array):
            result[f"gene_sentence_{top_n}"] = " ".join(all_genes_sorted_array[:top_n])
        else:
            result[f"gene_sentence_{top_n}"] = result["gene_sentence_all"]

    if gene_lists:
        # Keep Python comprehension with set membership but iterate numpy array directly
        for filestem, gene_set in gene_lists.items():
            filtered = [g for g in all_genes_sorted_array if g in gene_set]
            result[f"gene_sentence_{filestem}"] = " ".join(filtered)
    return result


def _generate_synthetic_data(
    n_genes: int,
    ensembl_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    expr = rng.random(n_genes, dtype=np.float64)

    n_ens = int(math.floor(n_genes * ensembl_fraction))
    n_sym = n_genes - n_ens

    # Generate Ensembl-like IDs (start with ENS)
    ens = np.array([f"ENSG{1000000+i:011d}" for i in range(n_ens)], dtype=object)
    # Generate simple gene symbols
    sym = np.array([f"GENE{i:05d}" for i in range(n_sym)], dtype=object)

    all_genes = np.concatenate([ens, sym])
    rng.shuffle(all_genes)
    return expr, all_genes


def _generate_gene_lists(
    all_symbols: np.ndarray,
    n_lists: int,
    list_sizes: list[int],
    seed: int,
) -> dict[str, set[str]]:
    rng = np.random.default_rng(seed + 42)
    # Keep only non-Ensembl symbols for lists (simulate real sets of symbols)
    mask = ~np.char.startswith(all_symbols.astype(str), "ENS")
    symbols = all_symbols[mask]
    lists: dict[str, set[str]] = {}
    for i in range(n_lists):
        size = list_sizes[min(i, len(list_sizes) - 1)]
        if size <= 0 or symbols.size == 0:
            lists[f"list{i+1}"] = set()
            continue
        idx = rng.choice(symbols.size, size=min(size, symbols.size), replace=False)
        lists[f"list{i+1}"] = set(map(str, symbols[idx].tolist()))
    return lists


def _time_call(
    fn: Callable[..., dict[str, str]],
    cell_expr: np.ndarray,
    gene_symbols: np.ndarray,
    top_n: Optional[int],
    gene_lists: Optional[dict[str, set[str]]],
    repeats: int,
) -> float:
    # Warmup
    fn(cell_expr, gene_symbols, top_n, gene_lists)
    start = time.perf_counter()
    for _ in range(repeats):
        fn(cell_expr, gene_symbols, top_n, gene_lists)
    end = time.perf_counter()
    return (end - start) / max(repeats, 1)


@app.command()
def run(
    n_genes: int = typer.Option(20000, "--n-genes", help="Number of genes"),
    ensembl_fraction: float = typer.Option(
        0.25, "--ens-frac", help="Fraction of Ensembl-like IDs"
    ),
    n_lists: int = typer.Option(3, "--n-lists", help="Number of gene lists"),
    list_sizes: str = typer.Option(
        "200,2000,5000",
        "--list-sizes",
        help="Comma-separated sizes for gene lists (cycled if fewer than n_lists)",
    ),
    top_n: Optional[int] = typer.Option(2000, "--top-n", help="Top-N for sentence"),
    repeats: int = typer.Option(5, "--repeats", help="Repeat each measurement"),
    seed: int = typer.Option(123, "--seed", help="Random seed"),
    log_file: Optional[Path] = typer.Option(
        None, "--log-file", "-l", help="Path to eliot rendered log file"
    ),
    log_stdout: bool = typer.Option(
        True, "--log-stdout/--no-log-stdout", help="Mirror Eliot logs to stdout"
    ),
) -> None:
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        json_path = log_file.with_suffix(".json")
        to_nice_file(output_file=json_path, rendered_file=log_file)
        if log_stdout:
            to_nice_stdout(output_file=json_path)
    elif log_stdout:
        to_file(typer.get_text_stream("stdout"))

    lists = [int(x.strip()) for x in list_sizes.split(",") if x.strip()]

    with start_action(
        action_type="bench_create_cell_sentence",
        n_genes=n_genes,
        ensembl_fraction=ensembl_fraction,
        n_lists=n_lists,
        list_sizes=lists,
        top_n=top_n,
        repeats=repeats,
        seed=seed,
    ) as action:
        cell_expr, all_symbols = _generate_synthetic_data(
            n_genes=n_genes, ensembl_fraction=ensembl_fraction, seed=seed
        )
        gene_lists = _generate_gene_lists(
            all_symbols, n_lists=n_lists, list_sizes=lists, seed=seed
        )
        # Ensure dtype object to match production usage
        all_symbols = all_symbols.astype(object, copy=False)

        variants: Dict[str, Callable[..., dict[str, str]]] = {
            "baseline_current": create_cell_sentence,
            "np_isin_filter": variant_np_isin,
            "no_tolist_join": variant_join_no_tolist,
        }

        results: dict[str, Any] = {}

        for name, fn in variants.items():
            avg_sec = _time_call(
                fn=fn,
                cell_expr=cell_expr,
                gene_symbols=all_symbols,
                top_n=top_n,
                gene_lists=gene_lists,
                repeats=repeats,
            )
            results[name] = {"avg_seconds": avg_sec}
            action.log(message_type="benchmark_result", variant=name, avg_seconds=avg_sec)

        # Print concise summary
        typer.echo("\nBenchmark results (avg seconds per call):")
        for name, res in results.items():
            typer.echo(f"  - {name}: {res['avg_seconds']:.6f}s")
        typer.echo("")

        # Write JSON next to log_file if provided
        if log_file:
            json_out = log_file.with_name(log_file.stem + "_results.json")
            json_out.write_text(json.dumps(results, indent=2))
            action.log(message_type="benchmark_results_written", path=str(json_out))
            typer.echo(f"Results: {json_out}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()


