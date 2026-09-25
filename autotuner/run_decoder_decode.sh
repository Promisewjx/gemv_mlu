#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/build}"
APP="${APP:-${BUILD_DIR}/bin/decoder_decode}"
OUT_DIR="${OUT_DIR:-${BUILD_DIR}/app}"
REPEAT="${REPEAT:-3}"
TOKENS="${TOKENS:-1000}"
WARMUP="${WARMUP:-10}"
HIDDEN="${HIDDEN:-256}"
INTERMEDIATE="${INTERMEDIATE:-512}"
LAYERS="${LAYERS:-1}"
SEED="${SEED:-20260925}"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    cat <<EOF
Usage: $0

Environment overrides:
  BUILD_DIR, OUT_DIR, REPEAT, TOKENS, WARMUP, HIDDEN, INTERMEDIATE, LAYERS, SEED
EOF
    exit 0
fi

if [[ ! -x "${APP}" ]]; then
    echo "Decoder application does not exist: ${APP}" >&2
    echo "Build with: cmake -S ${ROOT_DIR} -B ${BUILD_DIR} -DBUILD_APPS=ON && cmake --build ${BUILD_DIR} -j" >&2
    exit 2
fi

mkdir -p "${OUT_DIR}"

"${APP}" \
    --impl all \
    --backend all \
    --hidden "${HIDDEN}" \
    --intermediate "${INTERMEDIATE}" \
    --layers "${LAYERS}" \
    --tokens "${TOKENS}" \
    --warmup "${WARMUP}" \
    --repeat "${REPEAT}" \
    --seed "${SEED}" \
    --report "${OUT_DIR}/decoder_kernel_compare.md" \
    --csv "${OUT_DIR}/decoder_kernel_compare.csv"

echo "Generated:"
echo "- ${OUT_DIR}/decoder_kernel_compare.md"
echo "- ${OUT_DIR}/decoder_kernel_compare.csv"
