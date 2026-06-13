#!/usr/bin/env python3
import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


TEST_HEADER_RE = re.compile(r"=== 测试\s+(\d+):\s+(.+?)\s+===")
SHAPE_RE = re.compile(r"矩阵规模:\s*(\d+)×(\d+),\s*转置:\s*(是|否),\s*LDA倍数:\s*(\d+),\s*incx:\s*(\d+),\s*incy:\s*(\d+)")
TIME_RE = re.compile(r"Kernel耗时:\s*([0-9]*\.?[0-9]+)\s*ms")
BW_RE = re.compile(r"带宽:\s*([0-9]*\.?[0-9]+)\s*KB/s")
PASS_RE = re.compile(r"✅")
FAIL_RE = re.compile(r"❌")


@dataclass
class CaseResult:
    case_id: int
    case_name: str
    m: int
    n: int
    trans: str
    lda_factor: int
    incx: int
    incy: int
    time_ms: float
    bandwidth_kbs: float
    passed: bool


def parse_log(path: Path) -> Dict[int, CaseResult]:
    lines = path.read_text(encoding="utf-8").splitlines()
    results: Dict[int, CaseResult] = {}
    i = 0
    while i < len(lines):
        header_m = TEST_HEADER_RE.search(lines[i])
        if not header_m:
            i += 1
            continue

        case_id = int(header_m.group(1))
        case_name = header_m.group(2).strip()
        m = n = lda_factor = incx = incy = 0
        trans = ""
        time_ms = None
        bandwidth_kbs = None
        passed = False
        failed = False

        j = i + 1
        while j < len(lines):
            if TEST_HEADER_RE.search(lines[j]):
                break
            shape_m = SHAPE_RE.search(lines[j])
            if shape_m:
                m = int(shape_m.group(1))
                n = int(shape_m.group(2))
                trans = shape_m.group(3)
                lda_factor = int(shape_m.group(4))
                incx = int(shape_m.group(5))
                incy = int(shape_m.group(6))
            time_m = TIME_RE.search(lines[j])
            if time_m:
                time_ms = float(time_m.group(1))
            bw_m = BW_RE.search(lines[j])
            if bw_m:
                bandwidth_kbs = float(bw_m.group(1))
            if PASS_RE.search(lines[j]):
                passed = True
            if FAIL_RE.search(lines[j]) and "测试失败" in lines[j]:
                failed = True
            j += 1

        if time_ms is None:
            raise ValueError(f"{path}: case {case_id} missing kernel time")
        if bandwidth_kbs is None:
            raise ValueError(f"{path}: case {case_id} missing bandwidth")
        if m == 0 or n == 0:
            raise ValueError(f"{path}: case {case_id} missing shape")

        results[case_id] = CaseResult(
            case_id=case_id,
            case_name=case_name,
            m=m,
            n=n,
            trans=trans,
            lda_factor=lda_factor,
            incx=incx,
            incy=incy,
            time_ms=time_ms,
            bandwidth_kbs=bandwidth_kbs,
            passed=passed and not failed,
        )
        i = j
    return results


def geom_mean(values: List[float]) -> float:
    positive = [v for v in values if v > 0]
    if not positive:
        return float("nan")
    return math.exp(sum(math.log(v) for v in positive) / len(positive))


def percentile(values: List[float], p: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    pos = p * (len(s) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return s[lo]
    w = pos - lo
    return s[lo] * (1 - w) + s[hi] * w


def shape_bucket(m: int, n: int) -> str:
    if m == n:
        return "方阵"
    if m > n:
        return "高瘦矩阵"
    return "宽扁矩阵"


def avg(values: List[float]) -> float:
    return sum(values) / len(values)


def stdev(values: List[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mean = avg(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))


def group_value(c: CaseResult, key: str) -> str:
    if key == "shape_bucket":
        return shape_bucket(c.m, c.n)
    if key == "trans":
        return c.trans
    if key == "lda_factor":
        return f"lda_factor={c.lda_factor}"
    if key == "incx_incy":
        return f"incx={c.incx},incy={c.incy}"
    raise ValueError(key)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze repeated GEMV logs using averaged timings.")
    parser.add_argument("--logs", nargs="+", required=True, help="Format: impl=log1,log2,...")
    parser.add_argument("--out-dir", default="build/analysis_avg", help="Output directory")
    args = parser.parse_args()

    named_logs: List[Tuple[str, List[Path]]] = []
    for item in args.logs:
        if "=" not in item:
            raise ValueError(f"invalid --logs entry: {item}, expected impl=path1,path2")
        impl, paths_text = item.split("=", 1)
        paths = [Path(p).resolve() for p in paths_text.split(",") if p]
        if not paths:
            raise ValueError(f"impl {impl} has no paths")
        named_logs.append((impl.strip(), paths))

    if not named_logs:
        raise ValueError("no logs provided")

    parsed_repeats: Dict[str, List[Dict[int, CaseResult]]] = {}
    for impl, paths in named_logs:
        parsed_repeats[impl] = [parse_log(path) for path in paths]

    baseline_name = named_logs[0][0]
    baseline_first = parsed_repeats[baseline_name][0]
    case_ids = sorted(baseline_first.keys())

    for impl, repeats in parsed_repeats.items():
        for round_idx, cases in enumerate(repeats, 1):
            missing = [cid for cid in case_ids if cid not in cases]
            if missing:
                raise ValueError(f"impl {impl} round {round_idx} missing cases: {missing}")

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    avg_results: Dict[str, Dict[int, CaseResult]] = {}
    time_stdevs: Dict[Tuple[str, int], float] = {}
    pass_counts: Dict[Tuple[str, int], int] = {}

    for impl, repeats in parsed_repeats.items():
        avg_results[impl] = {}
        for cid in case_ids:
            first = repeats[0][cid]
            times = [cases[cid].time_ms for cases in repeats]
            bws = [cases[cid].bandwidth_kbs for cases in repeats]
            passes = [cases[cid].passed for cases in repeats]
            avg_results[impl][cid] = CaseResult(
                case_id=first.case_id,
                case_name=first.case_name,
                m=first.m,
                n=first.n,
                trans=first.trans,
                lda_factor=first.lda_factor,
                incx=first.incx,
                incy=first.incy,
                time_ms=avg(times),
                bandwidth_kbs=avg(bws),
                passed=all(passes),
            )
            time_stdevs[(impl, cid)] = stdev(times)
            pass_counts[(impl, cid)] = sum(1 for p in passes if p)

    baseline = avg_results[baseline_name]
    impl_order = [name for name, _ in named_logs]
    repeat_count = len(parsed_repeats[baseline_name])

    details_csv = out_dir / "gemv_speedup_details.csv"
    with details_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "impl",
            "case_id",
            "case_name",
            "shape_bucket",
            "m",
            "n",
            "trans",
            "lda_factor",
            "incx",
            "incy",
            "avg_time_ms",
            "time_stdev_ms",
            "avg_bandwidth_kbs",
            "speedup_vs_baseline",
            "pass_count",
            "repeat_count",
            "pass",
        ])
        for impl in impl_order:
            for cid in case_ids:
                c = avg_results[impl][cid]
                b = baseline[cid]
                speedup = b.time_ms / c.time_ms
                writer.writerow([
                    impl,
                    c.case_id,
                    c.case_name,
                    shape_bucket(c.m, c.n),
                    c.m,
                    c.n,
                    c.trans,
                    c.lda_factor,
                    c.incx,
                    c.incy,
                    f"{c.time_ms:.6f}",
                    f"{time_stdevs[(impl, cid)]:.6f}",
                    f"{c.bandwidth_kbs:.6f}",
                    f"{speedup:.6f}",
                    pass_counts[(impl, cid)],
                    len(parsed_repeats[impl]),
                    "PASS" if c.passed else "FAIL",
                ])

    summary_md = out_dir / "gemv_speedup_summary.md"
    group_keys = ["shape_bucket", "trans", "lda_factor", "incx_incy"]

    with summary_md.open("w", encoding="utf-8") as f:
        f.write("# GEMV Averaged Speedup Summary\n\n")
        f.write(f"- Baseline: `{baseline_name}`\n")
        f.write(f"- Cases: {len(case_ids)}\n")
        f.write(f"- Repeats: {repeat_count}\n")
        f.write(f"- Details CSV: `{details_csv}`\n\n")

        f.write("## Overall\n\n")
        f.write("| impl | pass_count | gmean_speedup | p50 | p90 | min | max |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for impl in impl_order:
            speeds = [baseline[cid].time_ms / avg_results[impl][cid].time_ms for cid in case_ids]
            pass_count = sum(1 for cid in case_ids if avg_results[impl][cid].passed)
            f.write(
                f"| {impl} | {pass_count}/{len(case_ids)} | {geom_mean(speeds):.4f} | "
                f"{percentile(speeds, 0.5):.4f} | {percentile(speeds, 0.9):.4f} | "
                f"{min(speeds):.4f} | {max(speeds):.4f} |\n"
            )
        f.write("\n")

        for gk in group_keys:
            f.write(f"## Grouped by {gk}\n\n")
            values = sorted({group_value(baseline[cid], gk) for cid in case_ids})
            f.write("| group | impl | gmean_speedup | avg_speedup | cases |\n")
            f.write("|---|---|---:|---:|---:|\n")
            for gv in values:
                member_ids = [cid for cid in case_ids if group_value(baseline[cid], gk) == gv]
                for impl in impl_order:
                    speeds = [baseline[cid].time_ms / avg_results[impl][cid].time_ms for cid in member_ids]
                    f.write(
                        f"| {gv} | {impl} | {geom_mean(speeds):.4f} | "
                        f"{sum(speeds) / len(speeds):.4f} | {len(member_ids)} |\n"
                    )
            f.write("\n")

    print(f"Generated averaged analysis:\n- {details_csv}\n- {summary_md}")


if __name__ == "__main__":
    main()
