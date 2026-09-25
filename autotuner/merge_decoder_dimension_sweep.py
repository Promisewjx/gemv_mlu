#!/usr/bin/env python3
"""Merge decoder all-kernel and BO CSV reports into final comparison tables."""

import argparse
import csv
from pathlib import Path


CONFIGS = (
    ("small", 256, 1024),
    ("medium", 512, 2048),
    ("large", 1024, 4096),
)


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    if not rows:
        raise RuntimeError("no decoder rows to write")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path, name, hidden, intermediate, rows, all_path, bo_path):
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"# Decoder Dimension Comparison: {name}\n\n")
        handle.write(f"- hidden: `{hidden}`\n")
        handle.write(f"- intermediate: `{intermediate}`\n")
        handle.write(f"- all-kernel source: `{all_path}`\n")
        handle.write(f"- BO source: `{bo_path}`\n\n")
        handle.write("| backend | verify | ms/token | speedup | GEMV % | attention % | other % |\n")
        handle.write("|---|---|---:|---:|---:|---:|---:|\n")
        for row in rows:
            handle.write(
                f"| `{row['backend']}` | {row['verify']} | "
                f"{float(row['ms_per_token']):.6f} | "
                f"{float(row['speedup_vs_baseline']):.4f} | "
                f"{float(row['gemv_pct']):.2f} | "
                f"{float(row['attention_pct']):.2f} | "
                f"{float(row['other_pct']):.2f} |\n"
            )


def merge_one(out_dir, name, hidden, intermediate):
    all_path = out_dir / f"{name}_all_kernels.csv"
    bo_path = out_dir / f"{name}_tile_sram_db_bo_best.csv"
    all_rows = read_rows(all_path)
    bo_rows = read_rows(bo_path)
    if not all_rows:
        raise RuntimeError(f"empty all-kernel report: {all_path}")
    if len(bo_rows) != 1:
        raise RuntimeError(f"expected one BO row in {bo_path}, got {len(bo_rows)}")

    baseline_ms = float(all_rows[0]["ms_per_token"])
    rows = [dict(row) for row in all_rows]
    bo_row = dict(bo_rows[0])
    bo_row["backend"] = "mlu_tile_sram_db_bo_best"
    bo_row["speedup_vs_baseline"] = f"{baseline_ms / float(bo_row['ms_per_token']):.6f}"
    rows.append(bo_row)

    output_csv = out_dir / f"{name}_final_compare.csv"
    output_md = out_dir / f"{name}_final_compare.md"
    write_csv(output_csv, rows)
    write_markdown(output_md, name, hidden, intermediate, rows, all_path, bo_path)
    return output_csv, output_md


def main():
    parser = argparse.ArgumentParser(description="Merge decoder dimension sweep reports.")
    parser.add_argument("--out-dir", default="build/decoder_dimension_sweep")
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    generated = []
    for name, hidden, intermediate in CONFIGS:
        generated.extend(merge_one(out_dir, name, hidden, intermediate))

    print("Generated decoder final comparison tables:")
    for path in generated:
        print(f"- {path}")


if __name__ == "__main__":
    main()
