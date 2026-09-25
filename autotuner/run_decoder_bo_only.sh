#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/build/decoder_bo_best}"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/build/app_decoder_bo_only}"
REPEAT="${REPEAT:-3}"
TOKENS="${TOKENS:-1000}"
WARMUP="${WARMUP:-10}"
HIDDEN="${HIDDEN:-256}"
INTERMEDIATE="${INTERMEDIATE:-512}"
LAYERS="${LAYERS:-1}"
SEED="${SEED:-20260925}"
MLU_ARCH="${MLU_ARCH:-270}"
JOBS="${JOBS:-4}"

# Best configuration recorded by the existing tile_sram_db Bayesian tuner.
BO_NRAM="${BO_NRAM:-1024}"
BO_ROWS="${BO_ROWS:-16}"
BO_UNROLL="${BO_UNROLL:-2}"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    cat <<EOF
Run only the Bayesian-optimized tile_sram_db decoder configuration.

BO parameters:
  NRAM_CHUNK_FLOATS=${BO_NRAM}
  TILE_SRAM_BLOCK_ROWS=${BO_ROWS}
  UNROLL_FACTOR=${BO_UNROLL}

Environment overrides:
  BUILD_DIR OUT_DIR REPEAT TOKENS WARMUP HIDDEN INTERMEDIATE LAYERS
  SEED MLU_ARCH JOBS BO_NRAM BO_ROWS BO_UNROLL
EOF
    exit 0
fi

cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" \
    -DHAVE_MLU=ON \
    -DMLU_ARCH="${MLU_ARCH}" \
    -DBUILD_APPS=ON \
    -DGEMV_ENABLE_CNBLAS=OFF \
    -DGEMV_NRAM_CHUNK_FLOATS="${BO_NRAM}" \
    -DGEMV_TILE_SRAM_BLOCK_ROWS="${BO_ROWS}" \
    -DGEMV_UNROLL_FACTOR="${BO_UNROLL}"
cmake --build "${BUILD_DIR}" -j"${JOBS}"

mkdir -p "${OUT_DIR}"
"${BUILD_DIR}/bin/decoder_decode" \
    --impl tile_sram_db \
    --backend mlu \
    --hidden "${HIDDEN}" \
    --intermediate "${INTERMEDIATE}" \
    --layers "${LAYERS}" \
    --tokens "${TOKENS}" \
    --warmup "${WARMUP}" \
    --repeat "${REPEAT}" \
    --seed "${SEED}" \
    --report "${OUT_DIR}/decoder_tile_sram_db_bo_best.md" \
    --csv "${OUT_DIR}/decoder_tile_sram_db_bo_best.csv"

echo "Generated:"
echo "- ${OUT_DIR}/decoder_tile_sram_db_bo_best.md"
echo "- ${OUT_DIR}/decoder_tile_sram_db_bo_best.csv"
