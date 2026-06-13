#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/build}"
TEST_BIN="${TEST_BIN:-${BUILD_DIR}/test/test_gemv}"
OUT_DIR="${OUT_DIR:-${BUILD_DIR}/perf_$(date +%Y%m%d_%H%M%S)}"
REPEAT=1
MODE="all"
SINGLE_TARGET=""

usage() {
    cat <<EOF
Usage: $0 [options]

Options:
  --repeat N        Run each implementation N times. Default: 1
  --out-dir DIR     Output directory. Default: build/perf_<timestamp>
  --quick           Run quick test mode instead of all 33 cases
  --single TARGET   Run one test by index or name keyword
  -h, --help        Show this help

Examples:
  $0
  $0 --repeat 3
  $0 --quick --repeat 5
  $0 --single 9
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --repeat)
            REPEAT="$2"
            shift 2
            ;;
        --out-dir)
            OUT_DIR="$2"
            shift 2
            ;;
        --quick)
            MODE="quick"
            shift
            ;;
        --single)
            MODE="single"
            SINGLE_TARGET="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if ! [[ "${REPEAT}" =~ ^[0-9]+$ ]] || [[ "${REPEAT}" -lt 1 ]]; then
    echo "--repeat must be a positive integer" >&2
    exit 2
fi

if [[ "${MODE}" == "single" && -z "${SINGLE_TARGET}" ]]; then
    echo "--single requires a test index or name keyword" >&2
    exit 2
fi

if [[ ! -d "${BUILD_DIR}" ]]; then
    echo "Build directory does not exist: ${BUILD_DIR}" >&2
    echo "Run cmake first, for example: mkdir -p build && cd build && cmake -DHAVE_MLU=ON -DMLU_ARCH=270 .." >&2
    exit 2
fi

echo "[build] make -C ${BUILD_DIR} -j"
make -C "${BUILD_DIR}" -j

if [[ ! -x "${TEST_BIN}" ]]; then
    echo "Test binary does not exist or is not executable: ${TEST_BIN}" >&2
    exit 2
fi

mkdir -p "${OUT_DIR}/logs"

test_args=()
case "${MODE}" in
    all)
        ;;
    quick)
        test_args=("quick")
        ;;
    single)
        test_args=("single" "${SINGLE_TARGET}")
        ;;
esac

impls=("baseline" "tile_nram" "tile_nram_db" "tile_sram" "tile_sram_db")

for ((round = 1; round <= REPEAT; ++round)); do
    round_logs=()
    echo
    echo "[round ${round}/${REPEAT}] mode=${MODE}"

    for impl in "${impls[@]}"; do
        log_path="${OUT_DIR}/logs/${impl}_r${round}.log"
        round_logs+=("${impl}=${log_path}")

        echo
        echo "[run] GEMV_IMPL=${impl} ${TEST_BIN} ${test_args[*]:-}"
        GEMV_IMPL="${impl}" "${TEST_BIN}" "${test_args[@]}" 2>&1 | tee "${log_path}"
    done

    analysis_dir="${OUT_DIR}/analysis_r${round}"
    echo
    echo "[analyze] ${analysis_dir}"
    python3 "${ROOT_DIR}/autotuner/analyze_gemv_logs.py" \
        --logs "${round_logs[@]}" \
        --out-dir "${analysis_dir}"
done

avg_logs=()
for impl in "${impls[@]}"; do
    impl_logs=()
    for ((round = 1; round <= REPEAT; ++round)); do
        impl_logs+=("${OUT_DIR}/logs/${impl}_r${round}.log")
    done
    joined_logs="$(IFS=,; echo "${impl_logs[*]}")"
    avg_logs+=("${impl}=${joined_logs}")
done

avg_analysis_dir="${OUT_DIR}/analysis_avg"
echo
echo "[analyze averaged repeats] ${avg_analysis_dir}"
python3 "${ROOT_DIR}/autotuner/analyze_gemv_repeats.py" \
    --logs "${avg_logs[@]}" \
    --out-dir "${avg_analysis_dir}"

echo
echo "Generated performance artifacts:"
echo "- Logs: ${OUT_DIR}/logs"
for ((round = 1; round <= REPEAT; ++round)); do
    echo "- Round ${round} summary: ${OUT_DIR}/analysis_r${round}/gemv_speedup_summary.md"
    echo "- Round ${round} details: ${OUT_DIR}/analysis_r${round}/gemv_speedup_details.csv"
done
echo "- Averaged summary: ${OUT_DIR}/analysis_avg/gemv_speedup_summary.md"
echo "- Averaged details: ${OUT_DIR}/analysis_avg/gemv_speedup_details.csv"
