#!/usr/bin/env python3
"""Print the best row from one or more decoder BO history files."""

import argparse
import csv
from pathlib import Path


def best_row(path):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"empty history: {path}")
    valid_rows = [
        row for row in rows
        if float(row["objective_gmean_ms_per_token"]) < 1e6
    ]
    if not valid_rows:
        raise RuntimeError(f"no valid evaluations in {path}")
    return min(valid_rows, key=lambda row: float(row["objective_gmean_ms_per_token"]))


def main():
    parser = argparse.ArgumentParser(description="Summarize decoder BO best parameters.")
    parser.add_argument("history", nargs="+", help="Path(s) to history.csv")
    args = parser.parse_args()

    best_rows = []
    for item in args.history:
        path = Path(item)
        row = best_row(path)
        best_rows.append(row)
        print(f"{path}:")
        print(f"  shapes={row['decoder_shapes']}")
        print(f"  NRAM={row['nram_chunk_floats']}")
        print(f"  ROWS={row['tile_sram_block_rows']}")
        print(f"  UNROLL={row['unroll_factor']}")
        print(f"  objective_ms_per_token={row['objective_gmean_ms_per_token']}")
        print(f"  shape_ms_per_token={row['shape_ms_per_token']}")

    if len(best_rows) > 1:
        print("\nEnvironment for run_decoder_dimension_sweep.sh:")
        print(
            "BO_NRAM_VALUES=\"{}\" \\".format(
                " ".join(row["nram_chunk_floats"] for row in best_rows)
            )
        )
        print(
            "BO_ROWS_VALUES=\"{}\" \\".format(
                " ".join(row["tile_sram_block_rows"] for row in best_rows)
            )
        )
        print(
            "BO_UNROLL_VALUES=\"{}\"".format(
                " ".join(row["unroll_factor"] for row in best_rows)
            )
        )


if __name__ == "__main__":
    main()
