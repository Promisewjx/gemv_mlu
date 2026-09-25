#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/build/gemv_final_compare}"
TEST_BIN="${TEST_BIN:-${BUILD_DIR}/test/test_gemv}"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/build/gemv_final_compare}"
REPEAT="${REPEAT:-3}"
MLU_ARCH="${MLU_ARCH:-270}"
JOBS="${JOBS:-4}"
MODE="all"
SINGLE_TARGET=""

DEFAULT_NRAM="${DEFAULT_NRAM:-256}"
DEFAULT_ROWS="${DEFAULT_ROWS:-16}"
DEFAULT_UNROLL="${DEFAULT_UNROLL:-1}"

BO_NRAM="${BO_NRAM:-1024}"
BO_ROWS="${BO_ROWS:-16}"
BO_UNROLL="${BO_UNROLL:-2}"

usage() {
    cat <<EOF
Usage: $0 [options]

Run one final GEMV kernel comparison table:
  baseline, tile_nram, tile_nram_db, tile_sram, tile_sram_db,
  tile_sram_xreuse, blas_style, tile_sram_db_bo_best, tile_sram_xreuse_bo_best.

Options:
  --repeat N          Repeats per implementation. Default: ${REPEAT}
  --out-dir DIR       Output directory. Default: ${OUT_DIR}
  --quick             Run quick test mode instead of all 33 cases
  --single TARGET     Run one test by index or name keyword
  --bo-nram N         BO NRAM_CHUNK_FLOATS. Default: ${BO_NRAM}
  --bo-rows N         BO TILE_SRAM_BLOCK_ROWS. Default: ${BO_ROWS}
  --bo-unroll N       BO UNROLL_FACTOR. Default: ${BO_UNROLL}
  -h, --help          Show this help
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
        --bo-nram)
            BO_NRAM="$2"
            shift 2
            ;;
        --bo-rows)
            BO_ROWS="$2"
            shift 2
            ;;
        --bo-unroll)
            BO_UNROLL="$2"
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

configure_build() {
    local nram="$1"
    local rows="$2"
    local unroll="$3"

    cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" \
        -DHAVE_MLU=ON \
        -DMLU_ARCH="${MLU_ARCH}" \
        -DBUILD_APPS=ON \
        -DGEMV_ENABLE_CNBLAS=OFF \
        -DGEMV_NRAM_CHUNK_FLOATS="${nram}" \
        -DGEMV_TILE_SRAM_BLOCK_ROWS="${rows}" \
        -DGEMV_UNROLL_FACTOR="${unroll}"
    cmake --build "${BUILD_DIR}" -j"${JOBS}"
}

run_impl_repeats() {
    local impl="$1"
    local label="$2"
    local nram="$3"
    local rows="$4"
    local unroll="$5"

    for ((round = 1; round <= REPEAT; ++round)); do
        local log_path="${OUT_DIR}/logs/${label}_r${round}.log"
        echo
        echo "[run] ${label} round ${round}/${REPEAT}: GEMV_IMPL=${impl}"
        echo "# impl: ${impl}" > "${log_path}"
        echo "# config: NRAM_CHUNK_FLOATS=${nram}, TILE_SRAM_BLOCK_ROWS=${rows}, UNROLL_FACTOR=${unroll}" >> "${log_path}"
        GEMV_IMPL="${impl}" "${TEST_BIN}" "${test_args[@]}" 2>&1 | tee -a "${log_path}"
    done
}

echo "[build] default kernel configuration"
configure_build "${DEFAULT_NRAM}" "${DEFAULT_ROWS}" "${DEFAULT_UNROLL}"

default_impls=("baseline" "tile_nram" "tile_nram_db" "tile_sram" "tile_sram_db" "tile_sram_xreuse" "blas_style")
for impl in "${default_impls[@]}"; do
    run_impl_repeats "${impl}" "${impl}" "${DEFAULT_NRAM}" "${DEFAULT_ROWS}" "${DEFAULT_UNROLL}"
done

echo
echo "[build] BO kernel configuration"
configure_build "${BO_NRAM}" "${BO_ROWS}" "${BO_UNROLL}"
run_impl_repeats "tile_sram_db" "tile_sram_db_bo_best" "${BO_NRAM}" "${BO_ROWS}" "${BO_UNROLL}"
run_impl_repeats "tile_sram_xreuse" "tile_sram_xreuse_bo_best" "${BO_NRAM}" "${BO_ROWS}" "${BO_UNROLL}"

labels=("baseline" "tile_nram" "tile_nram_db" "tile_sram" "tile_sram_db" "tile_sram_xreuse" "blas_style" "tile_sram_db_bo_best" "tile_sram_xreuse_bo_best")
avg_logs=()
for label in "${labels[@]}"; do
    logs=()
    for ((round = 1; round <= REPEAT; ++round)); do
        logs+=("${OUT_DIR}/logs/${label}_r${round}.log")
    done
    joined_logs="$(IFS=,; echo "${logs[*]}")"
    avg_logs+=("${label}=${joined_logs}")
done

avg_analysis_dir="${OUT_DIR}/analysis_avg"
echo
echo "[analyze averaged repeats] ${avg_analysis_dir}"
python3 "${ROOT_DIR}/autotuner/analyze_gemv_repeats.py" \
    --logs "${avg_logs[@]}" \
    --out-dir "${avg_analysis_dir}"

cat <<EOF

Generated final GEMV comparison:
  ${avg_analysis_dir}/gemv_speedup_summary.md
  ${avg_analysis_dir}/gemv_speedup_details.csv

Default build:
  NRAM=${DEFAULT_NRAM}, ROWS=${DEFAULT_ROWS}, UNROLL=${DEFAULT_UNROLL}
BO build:
  NRAM=${BO_NRAM}, ROWS=${BO_ROWS}, UNROLL=${BO_UNROLL}
EOF
