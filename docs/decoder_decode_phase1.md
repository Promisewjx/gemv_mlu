# Phase-One Decoder Decode Prototype

## Scope

This prototype models one or more decoder layers for autoregressive single-token decode.
Each layer contains:

1. Q/K/V projections
2. A single-head scaled dot-product attention reference path
3. Host-side KV-cache append and read
4. O projection
5. Residual connection
6. FFN up, gate, activation, down projections

Every projection is a row-major weight matrix evaluated through the existing
`BLAS_sGEMV(..., DnBLAS_OP_T, ...)` API. The runtime `GEMV_IMPL` environment variable therefore
selects the MLU implementation without changing application code.

The attention score, softmax, KV-cache storage, residual arithmetic and activation are host
reference code in this phase. This keeps the application small while still exposing the
repeated single-row GEMV workload that dominates a batch-size-one decode layer.

## Run

```bash
cmake -S . -B build -DHAVE_MLU=ON -DMLU_ARCH=270 -DBUILD_APPS=ON
cmake --build build -j

./build/bin/decoder_decode \
  --impl compare \
  --backend mlu \
  --hidden 256 \
  --intermediate 512 \
  --layers 1 \
  --tokens 1000 \
  --warmup 10 \
  --repeat 3 \
  --seed 20260925
```

`--impl compare` runs `baseline` and `tile_sram_db`; `--impl all` runs all seven MLU kernels
in the same process with identical deterministic weights and inputs:

```text
baseline
tile_nram
tile_nram_db
tile_sram
tile_sram_db
tile_sram_xreuse
blas_style
```

The output report averages repeated runs and includes:

- end-to-end milliseconds per token
- speedup relative to the first backend, normally `baseline`
- GEMV, attention and other time shares
- one-token CPU-reference verification
- configuration and CNRT version metadata

To run only one MLU implementation:

```bash
./build/bin/decoder_decode --impl tile_sram_db --backend mlu
```

For the full decode-kernel comparison:

```bash
./build/bin/decoder_decode \
  --impl all \
  --backend mlu \
  --hidden 256 \
  --intermediate 512 \
  --layers 1 \
  --tokens 1000 \
  --warmup 10 \
  --repeat 3 \
  --seed 20260925 \
  --report build/app/decoder_kernel_compare.md \
  --csv build/app/decoder_kernel_compare.csv
```

The all-kernel comparison uses the compile-time parameters of the selected build. To measure
the BO effect itself, build two independent configurations:

```bash
./autotuner/run_decoder_bo_compare.sh
```

The default configuration is `NRAM=256, ROWS=16, UNROLL=1`, while the recorded BO configuration
is `NRAM=1024, ROWS=16, UNROLL=2`. The reports contain these build parameters in their metadata.

If the all-kernel decoder report already exists, only run the additional BO configuration:

```bash
./autotuner/run_decoder_bo_only.sh
```

This runs only `tile_sram_db` with `NRAM=1024, ROWS=16, UNROLL=2`, so its output can be added
as the eighth row in the decoder kernel comparison table.

## Dimension sweep

To compare kernel behavior across small, medium and large decoder dimensions:

```bash
./autotuner/run_decoder_dimension_sweep.sh
```

The default sweep is:

```text
hidden=256,  intermediate=1024
hidden=512,  intermediate=2048
hidden=1024, intermediate=4096
```

It runs all seven kernels with the default build parameters and separately runs
`tile_sram_db` with the BO parameters. The default `tokens=100` keeps the CPU reference
attention from dominating the experiment; use `TOKENS=200` or `TOKENS=1000` for a longer
stability run after the initial comparison.

## Decoder-specific Bayesian tuning

The existing `tune_gemv.py` optimizes the 33-case GEMV microbenchmark. That objective is useful
for a general GEMV kernel, but it is not the same as optimizing decoder latency. For a
decoder-specific experiment, run:

```bash
python3 autotuner/tune_decoder.py \
  --iterations 20 \
  --repeat 1 \
  --tokens 100 \
  --warmup 10 \
  --out-dir build/bo_decoder
```

The default objective is the geometric mean of `ms/token` over:

```text
hidden=256,  intermediate=1024
hidden=512,  intermediate=2048
hidden=1024, intermediate=4096
```

The tuner searches the compile-time parameters of `tile_sram_db`. It writes `history.csv`,
per-evaluation reports under `logs/`, and a convergence PDF. This is an application-specific
tuning experiment and should be reported separately from the original GEMV microbenchmark BO.

The convenience wrapper is:

```bash
./autotuner/run_decoder_decode.sh
```

## cnBLAS status

cnBLAS support is guarded by `GEMV_ENABLE_CNBLAS`. It requires both `cnblas.h` and a
`libcnblas` library under `NEUWARE_HOME`. The current Neuware installation was inspected and
does not contain either artifact, so the default build reports cnBLAS as unavailable instead
of silently claiming a comparison.

When available:

```bash
cmake -S . -B build \
  -DHAVE_MLU=ON \
  -DMLU_ARCH=270 \
  -DBUILD_APPS=ON \
  -DGEMV_ENABLE_CNBLAS=ON
cmake --build build -j
./build/bin/decoder_decode --impl compare --backend all --repeat 3
```

## Prefill limitation

This repository currently contains a GEMV implementation but no GEMM implementation. The
phase-one application therefore measures decode directly and records the prefill comparison
as an explicit limitation. A true prefill-vs-decode experiment should add or link a GEMM
implementation and compare batch matrix projections against the single-row decode path.
