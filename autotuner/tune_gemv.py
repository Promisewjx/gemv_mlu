#!/usr/bin/env python3
import argparse
import csv
import datetime
import math
import os
import re
import subprocess
from pathlib import Path

import GPyOpt
from GPyOpt.methods import BayesianOptimization
import matplotlib.pyplot as plt


DEFAULT_EXCLUDE_CASES = "33"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Bayesian tune compile-time parameters for GEMV tile_sram_db."
    )
    parser.add_argument("--iterations", type=int, default=50, help="Bayesian optimization iterations")
    parser.add_argument("--repeat", type=int, default=1, help="Repeat test_gemv per parameter set")
    parser.add_argument(
        "--exclude-cases",
        default=DEFAULT_EXCLUDE_CASES,
        help="Comma-separated test case ids excluded from objective; default excludes 1x1 boundary case",
    )
    parser.add_argument("--include-boundary", action="store_true", help="Include all cases, including case 33")
    parser.add_argument("--out-dir", default="build/bo_tile_sram_db", help="Output directory")
    parser.add_argument("--quick", action="store_true", help="Use quick test mode for smoke tuning")
    return parser.parse_args()


def case_exclude_set(args):
    if args.include_boundary:
        return set()
    return {int(x) for x in args.exclude_cases.split(",") if x.strip()}


def parse_test_log(text):
    header_re = re.compile(r"=== 测试\s+(\d+):\s+(.+?)\s+===")
    time_re = re.compile(r"Kernel耗时:\s*([0-9]*\.?[0-9]+)\s*ms")
    pass_re = re.compile(r"✅")
    fail_re = re.compile(r"❌.*测试失败")

    cases = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        m = header_re.search(lines[i])
        if not m:
            i += 1
            continue
        case_id = int(m.group(1))
        case_name = m.group(2).strip()
        time_ms = None
        passed = False
        failed = False
        j = i + 1
        while j < len(lines) and not header_re.search(lines[j]):
            tm = time_re.search(lines[j])
            if tm:
                time_ms = float(tm.group(1))
            if pass_re.search(lines[j]):
                passed = True
            if fail_re.search(lines[j]):
                failed = True
            j += 1
        if time_ms is not None:
            cases.append({
                "case_id": case_id,
                "case_name": case_name,
                "time_ms": time_ms,
                "passed": passed and not failed,
            })
        i = j
    return cases


def geom_mean(values):
    positives = [v for v in values if v > 0]
    if not positives:
        return 0.0
    return math.exp(sum(math.log(v) for v in positives) / len(positives))


class TileSramDbTuner:
    def __init__(self, args):
        self.args = args
        self.script_dir = Path(__file__).resolve().parent
        self.project_root = self.script_dir.parent
        self.build_dir = self.project_root / "build"
        self.test_bin = self.build_dir / "test" / "test_gemv"
        self.out_dir = (self.project_root / args.out_dir).resolve()
        self.log_dir = self.out_dir / "logs"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.exclude_cases = case_exclude_set(args)
        self.eval_index = 0
        self.history_csv = self.out_dir / "history.csv"
        with self.history_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "eval",
                "nram_chunk_floats",
                "tile_sram_block_rows",
                "unroll_factor",
                "repeat",
                "objective_gmean_time_ms",
                "gmean_time_ms",
                "pass_count",
                "case_count",
                "excluded_cases",
            ])

    def configure_and_build(self, nram_chunk, tile_rows, unroll):
        cmake_cmd = [
            "cmake",
            "..",
            f"-DGEMV_NRAM_CHUNK_FLOATS={nram_chunk}",
            f"-DGEMV_TILE_SRAM_BLOCK_ROWS={tile_rows}",
            f"-DGEMV_UNROLL_FACTOR={unroll}",
        ]
        make_cmd = ["make", "-j4"]
        subprocess.run(cmake_cmd, cwd=self.build_dir, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        subprocess.run(make_cmd, cwd=self.build_dir, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    def run_once(self, eval_id, repeat_id):
        cmd = [str(self.test_bin)]
        if self.args.quick:
            cmd.append("quick")
        env = os.environ.copy()
        env["GEMV_IMPL"] = "tile_sram_db"
        proc = subprocess.run(cmd, cwd=self.build_dir, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        log_path = self.log_dir / f"eval_{eval_id:03d}_r{repeat_id}.log"
        log_path.write_text(proc.stdout, encoding="utf-8")
        if proc.returncode != 0:
            print(proc.stdout)
            raise RuntimeError(f"test_gemv failed with code {proc.returncode}")
        return parse_test_log(proc.stdout)

    def evaluate(self, parameters):
        params = parameters[0]
        nram_chunk = int(params[0])
        tile_rows = int(params[1])
        unroll = int(params[2])
        self.eval_index += 1
        eval_id = self.eval_index
        print(
            f"\n[eval {eval_id}] NRAM_CHUNK_FLOATS={nram_chunk}, "
            f"TILE_SRAM_BLOCK_ROWS={tile_rows}, UNROLL_FACTOR={unroll}"
        )

        try:
            self.configure_and_build(nram_chunk, tile_rows, unroll)
            all_times = {}
            pass_counts = {}
            names = {}
            for r in range(1, self.args.repeat + 1):
                cases = self.run_once(eval_id, r)
                for c in cases:
                    cid = c["case_id"]
                    if cid in self.exclude_cases:
                        continue
                    names[cid] = c["case_name"]
                    all_times.setdefault(cid, []).append(c["time_ms"])
                    pass_counts[cid] = pass_counts.get(cid, 0) + (1 if c["passed"] else 0)

            if not all_times:
                raise RuntimeError("no cases parsed for objective")
            failed = [cid for cid, cnt in pass_counts.items() if cnt != self.args.repeat]
            if failed:
                print(f"[eval {eval_id}] failed cases: {failed}")
                objective = 1e6
                gmean_time = 1e6
            else:
                avg_times = [sum(ts) / len(ts) for ts in all_times.values()]
                gmean_time = geom_mean(avg_times)
                objective = gmean_time
                print(
                    f"[eval {eval_id}] gmean_time_ms={gmean_time:.6f}, "
                    f"cases={len(avg_times)}, excluded={sorted(self.exclude_cases)}"
                )

            with self.history_csv.open("a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    eval_id,
                    nram_chunk,
                    tile_rows,
                    unroll,
                    self.args.repeat,
                    f"{objective:.9f}",
                    f"{gmean_time:.9f}",
                    sum(pass_counts.values()),
                    len(all_times) * self.args.repeat,
                    ";".join(str(x) for x in sorted(self.exclude_cases)),
                ])
            return objective
        except Exception as e:
            print(f"[eval {eval_id}] ERROR: {e}")
            with self.history_csv.open("a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    eval_id,
                    nram_chunk,
                    tile_rows,
                    unroll,
                    self.args.repeat,
                    "1000000.000000000",
                    "1000000.000000000",
                    0,
                    0,
                    ";".join(str(x) for x in sorted(self.exclude_cases)),
                ])
            return 1e6


def main():
    args = parse_args()
    tuner = TileSramDbTuner(args)

    domain = [
        {"name": "nram_chunk_floats", "type": "discrete", "domain": [64, 128, 256, 512, 1024]},
        {"name": "tile_sram_block_rows", "type": "discrete", "domain": [4, 8, 16, 32]},
        {"name": "unroll_factor", "type": "discrete", "domain": [1, 2, 4, 8]},
    ]

    optimizer = BayesianOptimization(
        f=tuner.evaluate,
        domain=domain,
        model_type="GP",
        acquisition_type="EI",
        acquisition_jitter=0.05,
        exact_feval=False,
        maximize=False,
    )

    print(f"--- Starting Bayesian Optimization for {args.iterations} iterations ---")
    print(f"Objective: minimize gmean_time_ms for GEMV_IMPL=tile_sram_db")
    print(f"Excluded cases from objective: {sorted(tuner.exclude_cases)}")
    optimizer.run_optimization(max_iter=args.iterations, verbosity=True)
    print("--- Bayesian Optimization Finished ---")

    best_params = {
        "NRAM_CHUNK_FLOATS": int(optimizer.x_opt[0]),
        "TILE_SRAM_BLOCK_ROWS": int(optimizer.x_opt[1]),
        "UNROLL_FACTOR": int(optimizer.x_opt[2]),
    }
    best_gmean_time = float(optimizer.fx_opt)

    print("\n=============================================")
    print("           Optimization Results")
    print("=============================================")
    print(f"Best objective gmean_time_ms: {best_gmean_time:.6f}")
    print("Best Parameters Found:")
    for name, value in best_params.items():
        print(f"  - {name}: {value}")
    print(f"History CSV: {tuner.history_csv}")
    print("=============================================\n")

    fig_name = tuner.out_dir / f"gemv_tile_sram_db_bo_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    optimizer.plot_convergence()
    plt.title("Convergence Plot of tile_sram_db GEMV Optimization")
    plt.ylabel("Best objective found (gmean_time_ms)")
    plt.tight_layout()
    plt.savefig(fig_name)
    print(f"Convergence plot saved to: {fig_name}")


if __name__ == "__main__":
    main()
