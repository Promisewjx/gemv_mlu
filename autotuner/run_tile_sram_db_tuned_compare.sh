#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/build}"
TEST_BIN="${TEST_BIN:-${BUILD_DIR}/test/test_gemv}"
OUT_DIR="${OUT_DIR:-${BUILD_DIR}/perf_tile_sram_db_tuned_$(date +%Y%m%d_%H%M%S)}"
REPEAT=3
MODE="all"
SINGLE_TARGET=""

BASE_NRAM=256
BASE_ROWS=16
BASE_UNROLL=1
TUNED_NRAM=1024
TUNED_ROWS=16
TUNED_UNROLL=2

usage() {
    cat <<EOF
Usage: $0 [options]

Compare plain tile_sram_db parameters against Bayesian-optimized parameters.

Options:
  --repeat N          Run each config N times. Default: 3
  --out-dir DIR       Output directory. Default: build/perf_tile_sram_db_tuned_<timestamp>
  --quick             Run quick test mode instead of all 33 cases
  --single TARGET     Run one test by index or name keyword
  --base-nram N       Baseline NRAM_CHUNK_FLOATS. Default: ${BASE_NRAM}
  --base-rows N       Baseline TILE_SRAM_BLOCK_ROWS. Default: ${BASE_ROWS}
  --base-unroll N     Baseline UNROLL_FACTOR. Default: ${BASE_UNROLL}
  --tuned-nram N      Tuned NRAM_CHUNK_FLOATS. Default: ${TUNED_NRAM}
  --tuned-rows N      Tuned TILE_SRAM_BLOCK_ROWS. Default: ${TUNED_ROWS}
  --tuned-unroll N    Tuned UNROLL_FACTOR. Default: ${TUNED_UNROLL}
  -h, --help          Show this help

Examples:
  $0 --repeat 3 --out-dir build/perf_tile_sram_db_tuned
  $0 --quick --repeat 3
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
        --base-nram)
            BASE_NRAM="$2"
            shift 2
            ;;
        --base-rows)
            BASE_ROWS="$2"
            shift 2
            ;;
        --base-unroll)
            BASE_UNROLL="$2"
            shift 2
            ;;
        --tuned-nram)
            TUNED_NRAM="$2"
            shift 2
            ;;
        --tuned-rows)
            TUNED_ROWS="$2"
            shift 2
            ;;
        --tuned-unroll)
            TUNED_UNROLL="$2"
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

mkdir -p "${OUT_DIR}/logs"

build_config() {
    local nram="$1"
    local rows="$2"
    local unroll="$3"
    echo "[build] NRAM_CHUNK_FLOATS=${nram}, TILE_SRAM_BLOCK_ROWS=${rows}, UNROLL_FACTOR=${unroll}"
    cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" \
        -DGEMV_NRAM_CHUNK_FLOATS="${nram}" \
        -DGEMV_TILE_SRAM_BLOCK_ROWS="${rows}" \
        -DGEMV_UNROLL_FACTOR="${unroll}" || return
    make -C "${BUILD_DIR}" -j || return
    if [[ ! -x "${TEST_BIN}" ]]; then
        echo "Test binary does not exist or is not executable: ${TEST_BIN}" >&2
        return 2
    fi
}

RESTORE_CONFIG_ON_EXIT=0

restore_default_config() {
    local status=$?
    if [[ "${RESTORE_CONFIG_ON_EXIT}" -eq 1 ]]; then
        RESTORE_CONFIG_ON_EXIT=0
        echo
        echo "[restore] default build config: NRAM_CHUNK_FLOATS=${BASE_NRAM}, TILE_SRAM_BLOCK_ROWS=${BASE_ROWS}, UNROLL_FACTOR=${BASE_UNROLL}"
        if ! build_config "${BASE_NRAM}" "${BASE_ROWS}" "${BASE_UNROLL}"; then
            echo "[restore] failed to restore default build config" >&2
        fi
    fi
    exit "${status}"
}

trap restore_default_config EXIT

run_config() {
    local name="$1"
    local nram="$2"
    local rows="$3"
    local unroll="$4"

    build_config "${nram}" "${rows}" "${unroll}"

    for ((round = 1; round <= REPEAT; ++round)); do
        local log_path="${OUT_DIR}/logs/${name}_r${round}.log"
        echo
        echo "[run] ${name} round ${round}/${REPEAT}: GEMV_IMPL=tile_sram_db ${TEST_BIN} ${test_args[*]:-}"
        echo "# config: NRAM_CHUNK_FLOATS=${nram}, TILE_SRAM_BLOCK_ROWS=${rows}, UNROLL_FACTOR=${unroll}" > "${log_path}"
        GEMV_IMPL=tile_sram_db "${TEST_BIN}" "${test_args[@]}" 2>&1 | tee -a "${log_path}"
    done
}

RESTORE_CONFIG_ON_EXIT=1
run_config "tile_sram_db_default" "${BASE_NRAM}" "${BASE_ROWS}" "${BASE_UNROLL}"
run_config "tile_sram_db_bo_best" "${TUNED_NRAM}" "${TUNED_ROWS}" "${TUNED_UNROLL}"

for ((round = 1; round <= REPEAT; ++round)); do
    analysis_dir="${OUT_DIR}/analysis_r${round}"
    echo
    echo "[analyze] ${analysis_dir}"
    python3 "${ROOT_DIR}/autotuner/analyze_gemv_logs.py" \
        --logs \
        "tile_sram_db_default=${OUT_DIR}/logs/tile_sram_db_default_r${round}.log" \
        "tile_sram_db_bo_best=${OUT_DIR}/logs/tile_sram_db_bo_best_r${round}.log" \
        --out-dir "${analysis_dir}"
done

default_logs=()
tuned_logs=()
for ((round = 1; round <= REPEAT; ++round)); do
    default_logs+=("${OUT_DIR}/logs/tile_sram_db_default_r${round}.log")
    tuned_logs+=("${OUT_DIR}/logs/tile_sram_db_bo_best_r${round}.log")
done
joined_default="$(IFS=,; echo "${default_logs[*]}")"
joined_tuned="$(IFS=,; echo "${tuned_logs[*]}")"

avg_analysis_dir="${OUT_DIR}/analysis_avg"
echo
echo "[analyze averaged repeats] ${avg_analysis_dir}"
python3 "${ROOT_DIR}/autotuner/analyze_gemv_repeats.py" \
    --logs \
    "tile_sram_db_default=${joined_default}" \
    "tile_sram_db_bo_best=${joined_tuned}" \
    --out-dir "${avg_analysis_dir}"

echo
echo "Generated tuned comparison artifacts:"
echo "- Logs: ${OUT_DIR}/logs"
for ((round = 1; round <= REPEAT; ++round)); do
    echo "- Round ${round} summary: ${OUT_DIR}/analysis_r${round}/gemv_speedup_summary.md"
    echo "- Round ${round} details: ${OUT_DIR}/analysis_r${round}/gemv_speedup_details.csv"
done
echo "- Averaged summary: ${OUT_DIR}/analysis_avg/gemv_speedup_summary.md"
echo "- Averaged details: ${OUT_DIR}/analysis_avg/gemv_speedup_details.csv"
