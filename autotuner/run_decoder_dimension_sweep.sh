#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/build/decoder_dimension_sweep}"
DEFAULT_BUILD="${DEFAULT_BUILD:-${ROOT_DIR}/build/decoder_sweep_default}"
BO_BUILD="${BO_BUILD:-${ROOT_DIR}/build/decoder_sweep_bo}"
MLU_ARCH="${MLU_ARCH:-270}"
JOBS="${JOBS:-4}"
TOKENS="${TOKENS:-100}"
WARMUP="${WARMUP:-10}"
REPEAT="${REPEAT:-3}"
SEED="${SEED:-20260925}"
LAYERS="${LAYERS:-1}"

# Default build parameters used by the seven-kernel comparison.
DEFAULT_NRAM="${DEFAULT_NRAM:-256}"
DEFAULT_ROWS="${DEFAULT_ROWS:-16}"
DEFAULT_UNROLL="${DEFAULT_UNROLL:-1}"

# Decoder-workload Bayesian-optimized tile_sram_db parameters. Use BO_*_VALUES
# to provide one parameter set per shape.
BO_NRAM="${BO_NRAM:-1024}"
BO_ROWS="${BO_ROWS:-16}"
BO_UNROLL="${BO_UNROLL:-1}"

# Small, medium and large reduction dimensions.
CONFIG_NAMES=("small" "medium" "large")
HIDDEN_VALUES=(${HIDDEN_VALUES:-256 512 1024})
INTERMEDIATE_VALUES=(${INTERMEDIATE_VALUES:-1024 2048 4096})
if [[ -n "${BO_NRAM_VALUES:-}" ]]; then
    BO_NRAM_LIST=(${BO_NRAM_VALUES})
else
    BO_NRAM_LIST=("${BO_NRAM}" "${BO_NRAM}" "${BO_NRAM}")
fi
if [[ -n "${BO_ROWS_VALUES:-}" ]]; then
    BO_ROWS_LIST=(${BO_ROWS_VALUES})
else
    BO_ROWS_LIST=("${BO_ROWS}" "${BO_ROWS}" "${BO_ROWS}")
fi
if [[ -n "${BO_UNROLL_VALUES:-}" ]]; then
    BO_UNROLL_LIST=(${BO_UNROLL_VALUES})
else
    BO_UNROLL_LIST=("${BO_UNROLL}" "${BO_UNROLL}" "${BO_UNROLL}")
fi

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    cat <<EOF
Run decoder GEMV kernel comparisons over multiple hidden/intermediate dimensions.

Default configurations:
  small:  hidden=256,  intermediate=1024
  medium: hidden=512,  intermediate=2048
  large:  hidden=1024, intermediate=4096

Each configuration runs:
  1. all seven MLU kernels with default build parameters
  2. tile_sram_db with BO parameters

Defaults:
  tokens=${TOKENS}, warmup=${WARMUP}, repeat=${REPEAT}, layers=${LAYERS}

Environment overrides:
  OUT_DIR DEFAULT_BUILD BO_BUILD MLU_ARCH JOBS TOKENS WARMUP REPEAT SEED LAYERS
  HIDDEN_VALUES INTERMEDIATE_VALUES
  DEFAULT_NRAM DEFAULT_ROWS DEFAULT_UNROLL
  BO_NRAM BO_ROWS BO_UNROLL
  BO_NRAM_VALUES BO_ROWS_VALUES BO_UNROLL_VALUES
EOF
    exit 0
fi

if [[ "${#HIDDEN_VALUES[@]}" -ne "${#CONFIG_NAMES[@]}" ||
      "${#INTERMEDIATE_VALUES[@]}" -ne "${#CONFIG_NAMES[@]}" ||
      "${#BO_NRAM_LIST[@]}" -ne "${#CONFIG_NAMES[@]}" ||
      "${#BO_ROWS_LIST[@]}" -ne "${#CONFIG_NAMES[@]}" ||
      "${#BO_UNROLL_LIST[@]}" -ne "${#CONFIG_NAMES[@]}" ]]; then
    echo "HIDDEN_VALUES, INTERMEDIATE_VALUES, and BO_*_VALUES must each contain 3 values" >&2
    exit 2
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

run_all_kernels() {
    local name="$1"
    local hidden="$2"
    local intermediate="$3"

    "${DEFAULT_BUILD}/bin/decoder_decode" \
        --impl all \
        --backend mlu \
        --hidden "${hidden}" \
        --intermediate "${intermediate}" \
        --layers "${LAYERS}" \
        --tokens "${TOKENS}" \
        --warmup "${WARMUP}" \
        --repeat "${REPEAT}" \
        --seed "${SEED}" \
        --report "${OUT_DIR}/${name}_all_kernels.md" \
        --csv "${OUT_DIR}/${name}_all_kernels.csv"
}

run_bo_kernel() {
    local name="$1"
    local hidden="$2"
    local intermediate="$3"

    "${BO_BUILD}/bin/decoder_decode" \
        --impl tile_sram_db \
        --backend mlu \
        --hidden "${hidden}" \
        --intermediate "${intermediate}" \
        --layers "${LAYERS}" \
        --tokens "${TOKENS}" \
        --warmup "${WARMUP}" \
        --repeat "${REPEAT}" \
        --seed "${SEED}" \
        --report "${OUT_DIR}/${name}_tile_sram_db_bo_best.md" \
        --csv "${OUT_DIR}/${name}_tile_sram_db_bo_best.csv"
}

mkdir -p "${OUT_DIR}"

echo "[build] default seven-kernel configuration"
configure_build "${DEFAULT_BUILD}" "${DEFAULT_NRAM}" "${DEFAULT_ROWS}" "${DEFAULT_UNROLL}"

for index in "${!CONFIG_NAMES[@]}"; do
    name="${CONFIG_NAMES[index]}"
    hidden="${HIDDEN_VALUES[index]}"
    intermediate="${INTERMEDIATE_VALUES[index]}"
    bo_nram="${BO_NRAM_LIST[index]}"
    bo_rows="${BO_ROWS_LIST[index]}"
    bo_unroll="${BO_UNROLL_LIST[index]}"

    echo
    echo "[case] ${name}: hidden=${hidden}, intermediate=${intermediate}"
    run_all_kernels "${name}" "${hidden}" "${intermediate}"

    echo "[build] BO tile_sram_db configuration for ${name}: NRAM=${bo_nram}, ROWS=${bo_rows}, UNROLL=${bo_unroll}"
    configure_build "${BO_BUILD}" "${bo_nram}" "${bo_rows}" "${bo_unroll}"
    run_bo_kernel "${name}" "${hidden}" "${intermediate}"
done

python3 "${ROOT_DIR}/autotuner/merge_decoder_dimension_sweep.py" \
    --out-dir "${OUT_DIR}"

cat <<EOF

Generated reports under:
  ${OUT_DIR}

Each dimension has:
  <name>_all_kernels.md
  <name>_all_kernels.csv
  <name>_tile_sram_db_bo_best.md
  <name>_tile_sram_db_bo_best.csv
  <name>_final_compare.md
  <name>_final_compare.csv

Default GEMV build:
  NRAM=${DEFAULT_NRAM}, ROWS=${DEFAULT_ROWS}, UNROLL=${DEFAULT_UNROLL}
BO build:
  NRAM_VALUES=${BO_NRAM_LIST[*]}
  ROWS_VALUES=${BO_ROWS_LIST[*]}
  UNROLL_VALUES=${BO_UNROLL_LIST[*]}
EOF
