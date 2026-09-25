#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_ROOT="${BUILD_ROOT:-${ROOT_DIR}/build/decoder_configs}"
DEFAULT_BUILD="${DEFAULT_BUILD:-${BUILD_ROOT}/default}"
TUNED_BUILD="${TUNED_BUILD:-${BUILD_ROOT}/bo_best}"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/build/app_decoder_bo_compare}"
REPEAT="${REPEAT:-3}"
TOKENS="${TOKENS:-1000}"
WARMUP="${WARMUP:-10}"
HIDDEN="${HIDDEN:-256}"
INTERMEDIATE="${INTERMEDIATE:-512}"
LAYERS="${LAYERS:-1}"
SEED="${SEED:-20260925}"
MLU_ARCH="${MLU_ARCH:-270}"
JOBS="${JOBS:-4}"

DEFAULT_NRAM="${DEFAULT_NRAM:-256}"
DEFAULT_ROWS="${DEFAULT_ROWS:-16}"
DEFAULT_UNROLL="${DEFAULT_UNROLL:-1}"
TUNED_NRAM="${TUNED_NRAM:-1024}"
TUNED_ROWS="${TUNED_ROWS:-16}"
TUNED_UNROLL="${TUNED_UNROLL:-2}"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    cat <<EOF
Compare decoder latency using default and Bayesian-optimized GEMV build parameters.

Default configuration:
  NRAM=${DEFAULT_NRAM}, ROWS=${DEFAULT_ROWS}, UNROLL=${DEFAULT_UNROLL}
BO configuration:
  NRAM=${TUNED_NRAM}, ROWS=${TUNED_ROWS}, UNROLL=${TUNED_UNROLL}

Environment overrides:
  BUILD_ROOT DEFAULT_BUILD TUNED_BUILD OUT_DIR REPEAT TOKENS WARMUP
  HIDDEN INTERMEDIATE LAYERS SEED MLU_ARCH JOBS
EOF
    exit 0
fi

configure_build() {
    local build_dir="$1"
    local nram="$2"
    local rows="$3"
    local unroll="$4"
    cmake -S "${ROOT_DIR}" -B "${build_dir}" \
        -DHAVE_MLU=ON \
        -DMLU_ARCH="${MLU_ARCH}" \
        -DBUILD_APPS=ON \
        -DGEMV_ENABLE_CNBLAS=OFF \
        -DGEMV_NRAM_CHUNK_FLOATS="${nram}" \
        -DGEMV_TILE_SRAM_BLOCK_ROWS="${rows}" \
        -DGEMV_UNROLL_FACTOR="${unroll}"
    cmake --build "${build_dir}" -j"${JOBS}"
}

run_decode() {
    local label="$1"
    local build_dir="$2"
    mkdir -p "${OUT_DIR}"
    "${build_dir}/bin/decoder_decode" \
        --impl tile_sram_db \
        --backend mlu \
        --hidden "${HIDDEN}" \
        --intermediate "${INTERMEDIATE}" \
        --layers "${LAYERS}" \
        --tokens "${TOKENS}" \
        --warmup "${WARMUP}" \
        --repeat "${REPEAT}" \
        --seed "${SEED}" \
        --report "${OUT_DIR}/${label}.md" \
        --csv "${OUT_DIR}/${label}.csv"
}

echo "[build] default GEMV parameters"
configure_build "${DEFAULT_BUILD}" "${DEFAULT_NRAM}" "${DEFAULT_ROWS}" "${DEFAULT_UNROLL}"
echo "[run] default decoder"
run_decode "decoder_default" "${DEFAULT_BUILD}"

echo "[build] Bayesian-optimized GEMV parameters"
configure_build "${TUNED_BUILD}" "${TUNED_NRAM}" "${TUNED_ROWS}" "${TUNED_UNROLL}"
echo "[run] BO-tuned decoder"
run_decode "decoder_bo_best" "${TUNED_BUILD}"

cat <<EOF
Generated decoder reports:
  ${OUT_DIR}/decoder_default.md
  ${OUT_DIR}/decoder_bo_best.md
  ${OUT_DIR}/decoder_default.csv
  ${OUT_DIR}/decoder_bo_best.csv

Compare the tile_sram_db rows:
  default: NRAM=${DEFAULT_NRAM}, ROWS=${DEFAULT_ROWS}, UNROLL=${DEFAULT_UNROLL}
  BO best: NRAM=${TUNED_NRAM}, ROWS=${TUNED_ROWS}, UNROLL=${TUNED_UNROLL}
EOF
