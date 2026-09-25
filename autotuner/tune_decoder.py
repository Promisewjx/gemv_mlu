#!/usr/bin/env python3
"""Bayesian tune tile_sram_db against three decoder decode shapes."""

import argparse
import csv
import datetime
import math
import os
import subprocess
from pathlib import Path

from GPyOpt.methods import BayesianOptimization
import matplotlib.pyplot as plt


DEFAULT_SHAPES = ((256, 1024), (512, 2048), (1024, 4096))


def parse_int_list(text):
    values = []
    for item in text.split(","):
        item = item.strip()
        if item:
            values.append(int(item))
    if not values:
        raise ValueError("empty integer list")
    return values


def parse_args():
    parser = argparse.ArgumentParser(
        description="Tune compile-time GEMV parameters using decoder ms/token."
    )
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--tokens", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260925)
    parser.add_argument("--mlu-arch", default="270")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--build-dir", default="build/bo_decoder")
    parser.add_argument("--out-dir", default="build/bo_decoder")
    parser.add_argument(
        "--shapes",
        default="256:1024,512:2048,1024:4096",
        help="Comma-separated hidden:intermediate pairs",
    )
    parser.add_argument(
        "--nram-domain",
        default="64,128,256,512,1024",
        help="Comma-separated NRAM_CHUNK_FLOATS candidates",
    )
    parser.add_argument(
        "--rows-domain",
        default="4,8,16",
        help=(
            "Comma-separated TILE_SRAM_BLOCK_ROWS candidates. The decoder "
            "default excludes 32 because it failed correctness on tested shapes."
        ),
    )
    parser.add_argument(
        "--unroll-domain",
        default="1,2,4,8",
        help="Comma-separated UNROLL_FACTOR candidates",
    )
    return parser.parse_args()


def parse_shapes(text):
    shapes = []
    for item in text.split(","):
        hidden, intermediate = item.split(":")
        shapes.append((int(hidden), int(intermediate)))
    if not shapes:
        raise ValueError("no decoder shapes")
    return shapes


def geom_mean(values):
    if not values or any(value <= 0 for value in values):
        return 1e6
    return math.exp(sum(math.log(value) for value in values) / len(values))


def read_report(path):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise RuntimeError(f"expected one backend row in {path}")
    row = rows[0]
    if row["verify"] != "PASS":
        raise RuntimeError(f"decoder verification failed in {path}")
    return float(row["ms_per_token"])


class DecoderTuner:
    def __init__(self, args, shapes):
        self.args = args
        self.shapes = shapes
        self.root = Path(__file__).resolve().parent.parent
        self.build_dir = (self.root / args.build_dir).resolve()
        self.out_dir = (self.root / args.out_dir).resolve()
        self.log_dir = self.out_dir / "logs"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.eval_id = 0
        self.cache = {}
        self.history = self.out_dir / "history.csv"
        with self.history.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow([
                "eval",
                "nram_chunk_floats",
                "tile_sram_block_rows",
                "unroll_factor",
                "repeat",
                "decoder_shapes",
                "objective_gmean_ms_per_token",
                "shape_ms_per_token",
            ])

    def configure_and_build(self, nram, rows, unroll):
        configure = [
            "cmake",
            "-S",
            str(self.root),
            "-B",
            str(self.build_dir),
            "-DHAVE_MLU=ON",
            f"-DMLU_ARCH={self.args.mlu_arch}",
            "-DBUILD_APPS=ON",
            "-DGEMV_ENABLE_CNBLAS=OFF",
            f"-DGEMV_NRAM_CHUNK_FLOATS={nram}",
            f"-DGEMV_TILE_SRAM_BLOCK_ROWS={rows}",
            f"-DGEMV_UNROLL_FACTOR={unroll}",
        ]
        subprocess.run(
            configure,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        subprocess.run(
            ["cmake", "--build", str(self.build_dir), f"-j{self.args.jobs}"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

    def evaluate(self, parameters):
        nram, rows, unroll = [int(value) for value in parameters[0]]
        self.eval_id += 1
        eval_dir = self.log_dir / f"eval_{self.eval_id:03d}"
        eval_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"[eval {self.eval_id}] NRAM={nram}, ROWS={rows}, UNROLL={unroll}"
        )

        key = (nram, rows, unroll)
        if key in self.cache:
            objective, shape_times = self.cache[key]
            print(f"[eval {self.eval_id}] cached objective={objective:.6f}")
            self.write_history(nram, rows, unroll, objective, shape_times)
            return objective

        shape_times = []
        try:
            self.configure_and_build(nram, rows, unroll)
            for shape_index, (hidden, intermediate) in enumerate(self.shapes):
                times = []
                for repeat in range(self.args.repeat):
                    report = eval_dir / f"shape_{shape_index}_repeat_{repeat}.csv"
                    command = [
                        str(self.build_dir / "bin/decoder_decode"),
                        "--impl",
                        "tile_sram_db",
                        "--backend",
                        "mlu",
                        "--hidden",
                        str(hidden),
                        "--intermediate",
                        str(intermediate),
                        "--layers",
                        str(self.args.layers),
                        "--tokens",
                        str(self.args.tokens),
                        "--warmup",
                        str(self.args.warmup),
                        "--repeat",
                        "1",
                        "--seed",
                        str(self.args.seed),
                        "--no-verify" if repeat else "",
                        "--report",
                        str(report.with_suffix(".md")),
                        "--csv",
                        str(report),
                    ]
                    command = [item for item in command if item]
                    environment = os.environ.copy()
                    environment["GEMV_IMPL"] = "tile_sram_db"
                    subprocess.run(
                        command,
                        cwd=self.root,
                        env=environment,
                        check=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                    )
                    times.append(read_report(report))
                shape_times.append(sum(times) / len(times))

            objective = geom_mean(shape_times)
            print(
                f"[eval {self.eval_id}] objective={objective:.6f}, "
                f"shape_times={[round(value, 6) for value in shape_times]}"
            )
        except Exception as error:
            print(f"[eval {self.eval_id}] failed: {error}")
            objective = 1e6
            shape_times = [1e6] * len(self.shapes)

        self.cache[key] = (objective, shape_times)
        self.write_history(nram, rows, unroll, objective, shape_times)
        return objective

    def write_history(self, nram, rows, unroll, objective, shape_times):
        with self.history.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow([
                self.eval_id,
                nram,
                rows,
                unroll,
                self.args.repeat,
                ";".join(f"{h}:{i}" for h, i in self.shapes),
                f"{objective:.9f}",
                ";".join(f"{value:.9f}" for value in shape_times),
            ])


def main():
    args = parse_args()
    shapes = parse_shapes(args.shapes)
    nram_domain = parse_int_list(args.nram_domain)
    rows_domain = parse_int_list(args.rows_domain)
    unroll_domain = parse_int_list(args.unroll_domain)
    tuner = DecoderTuner(args, shapes)
    optimizer = BayesianOptimization(
        f=tuner.evaluate,
        domain=[
            {"name": "nram_chunk_floats", "type": "discrete", "domain": nram_domain},
            {"name": "tile_sram_block_rows", "type": "discrete", "domain": rows_domain},
            {"name": "unroll_factor", "type": "discrete", "domain": unroll_domain},
        ],
        model_type="GP",
        acquisition_type="EI",
        acquisition_jitter=0.05,
        exact_feval=False,
        maximize=False,
    )
    optimizer.run_optimization(max_iter=args.iterations, verbosity=True)

    best = {
        "NRAM_CHUNK_FLOATS": int(optimizer.x_opt[0]),
        "TILE_SRAM_BLOCK_ROWS": int(optimizer.x_opt[1]),
        "UNROLL_FACTOR": int(optimizer.x_opt[2]),
    }
    print("Best decoder configuration:")
    for key, value in best.items():
        print(f"  {key}={value}")
    print(f"Best objective gmean_ms_per_token={float(optimizer.fx_opt):.6f}")
    print(f"History CSV: {tuner.history}")

    figure = tuner.out_dir / (
        f"decoder_bo_convergence_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    )
    optimizer.plot_convergence()
    plt.title("Decoder GEMV Bayesian Optimization")
    plt.ylabel("Best geometric mean ms/token")
    plt.tight_layout()
    plt.savefig(figure)
    print(f"Convergence plot: {figure}")


if __name__ == "__main__":
    main()
