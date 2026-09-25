# GEMV MLU

A Cambricon MLU implementation and benchmark suite for single-precision GEMV (`sgemv`). The project includes host-side wrappers, multiple BANG C kernel variants, correctness tests, performance comparison scripts, and a Bayesian autotuner for compile-time kernel parameters.

## Features

- MLU `sgemv` host API and utility wrappers.
- Multiple kernel implementations for baseline, NRAM tiling, SRAM tiling, double buffering, and BLAS-style pipelining.
- Correctness test suite covering 33 GEMV cases.
- Performance scripts that run repeated benchmarks and generate Markdown/CSV reports.
- Bayesian autotuning for the current best `tile_sram_db` kernel.
- A minimal decoder-layer decode application that uses GEMV for single-token Q/K/V/O and FFN projections.
- Decoder-workload Bayesian autotuning across three hidden/intermediate dimensions.

## Repository Layout

```text
.
├── include/                    # Public headers
├── src/
│   ├── gemv.cpp                # Host API entry
│   ├── app/decoder_decode.cpp  # Single-token decoder decode prototype
│   ├── kernels/mlu/gemv.mlu    # MLU kernels
│   └── utils/                  # Memory and handle helpers
├── tests/test_gemv.cpp         # Correctness and timing tests
├── autotuner/                  # Benchmark and tuning scripts
└── CMakeLists.txt
```

## Build

```bash
mkdir -p build
cd build
cmake -DHAVE_MLU=ON -DMLU_ARCH=270 ..
make -j
```

Useful compile-time parameters:

```bash
cmake .. \
  -DGEMV_NRAM_CHUNK_FLOATS=1024 \
  -DGEMV_TILE_SRAM_BLOCK_ROWS=16 \
  -DGEMV_UNROLL_FACTOR=2
make -j
```

## Test

The test program contains 33 cases covering square, tall-skinny, wide, transposed, large-LDA, non-unit-stride, and `1x1` boundary inputs. The default seed is `20260613`; set `GEMV_SEED` to reproduce another dataset.

```bash
cd build

# Run all cases
./test/test_gemv

# Quick smoke test
./test/test_gemv quick

# List cases or run one case
./test/test_gemv list
./test/test_gemv single 9

# Run through CTest
ctest --output-on-failure
```

Select a kernel with `GEMV_IMPL`:

```bash
GEMV_IMPL=baseline     ./test/test_gemv
GEMV_IMPL=tile_nram    ./test/test_gemv
GEMV_IMPL=tile_nram_db ./test/test_gemv
GEMV_IMPL=tile_sram    ./test/test_gemv
GEMV_IMPL=tile_sram_db ./test/test_gemv
GEMV_IMPL=tile_sram_xreuse ./test/test_gemv
GEMV_IMPL=blas_style   ./test/test_gemv
```

If `GEMV_IMPL` is not set, the current host path defaults to `blas_style`.

## Kernel Variants

| Kernel | Description |
|---|---|
| `baseline` | Scalar baseline. Each task handles multiple output rows and reads data directly from GDRAM. |
| `tile_nram` | Tiles the reduction dimension into NRAM and uses vector multiply/reduce. |
| `tile_nram_db` | Adds NRAM ping-pong double buffering to `tile_nram`. |
| `tile_sram` | Tiles A through `GDRAM -> SRAM -> NRAM` to improve A movement and reuse. |
| `tile_sram_db` | Adds SRAM ping-pong double buffering to `tile_sram`. Current best manually designed kernel. |
| `tile_sram_xreuse` | Experimental `tile_sram_db` variant that stages x through SRAM before each core loads it into NRAM. |
| `blas_style` | BLAS-style prologue / steady-state / epilogue implementation based on the SRAM tiled path. |

## Benchmark

Run the main benchmark comparison:

```bash
cd /home/LCUDA/wjx/gemv_mlu
./autotuner/run_gemv_perf_compare.sh --repeat 3 --out-dir build/perf
```

The script benchmarks:

```text
baseline -> tile_nram -> tile_nram_db -> tile_sram -> tile_sram_db -> tile_sram_xreuse -> blas_style
```

Important outputs:

```text
build/perf/analysis_avg/gemv_speedup_summary.md
build/perf/analysis_avg/gemv_speedup_details.csv
```

`analysis_avg` is the preferred report because it averages repeated runs. In the summary, the first implementation is the baseline and `speedup = baseline_time / impl_time`.

Useful variants:

```bash
# Quick benchmark
./autotuner/run_gemv_perf_compare.sh --quick --repeat 3 --out-dir build/perf_quick

# Benchmark one case
./autotuner/run_gemv_perf_compare.sh --single 9 --repeat 3 --out-dir build/perf_case9
```

## Current Results

### GEMV Kernel Comparison

Main 33-case benchmark, repeated 3 times and averaged. The first seven rows are measured against
the scalar `baseline`. The BO rows use the tuned comparison reports and are normalized to the
same scalar baseline through the measured `tile_sram_db_default` speedup.

| Implementation | Pass | GMean Speedup | p50 | p90 | Min | Max |
|---|---:|---:|---:|---:|---:|---:|
| `baseline` | 33/33 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| `tile_nram` | 33/33 | 8.4158 | 5.0523 | 81.6053 | 0.7895 | 81.7812 |
| `tile_nram_db` | 33/33 | 9.3350 | 4.8202 | 145.4371 | 0.7143 | 154.0294 |
| `tile_sram` | 33/33 | 10.1134 | 5.6831 | 101.4254 | 0.6000 | 122.0473 |
| `tile_sram_db` | 33/33 | 10.1928 | 5.6344 | 110.3677 | 0.5556 | 129.8055 |
| `tile_sram_xreuse` | 33/33 | 12.0785 | 0.9902* | 2.5334* | 0.7068* | 2.7265* |
| `blas_style` | 33/33 | 9.6996 | 5.3419 | 105.1551 | 0.5000 | 129.0708 |
| `tile_sram_db_bo_best` | 33/33 | 11.5770 | 1.0707* | 1.4480* | 0.9000* | 1.8610* |
| `tile_sram_xreuse_bo_best` | 33/33 | 14.2771 | 1.0969* | 2.8743* | 0.9000* | 3.3696* |

Rows marked with `*` are from the tuned comparison where the baseline is `tile_sram_db_default`,
so those distribution statistics are relative to `tile_sram_db_default`; the GMean column has
been converted to the scalar-baseline scale. `tile_sram_xreuse_bo_best` is the strongest kernel
result in the current reports, with a geometric mean speedup of about `14.28x` over the scalar
baseline. Most of the gain before BO comes from tiling and using the `GDRAM -> SRAM -> NRAM`
path; the extra double-buffer steps provide smaller incremental gains.

## Autotuning

The Bayesian tuner searches compile-time parameters for `tile_sram_db`:

- `NRAM_CHUNK_FLOATS`
- `TILE_SRAM_BLOCK_ROWS`
- `UNROLL_FACTOR`

Default tuning excludes the `1x1` boundary case from the objective because it is dominated by fixed launch/synchronization overhead. It should still be included in final validation.

```bash
cd /home/LCUDA/wjx/gemv_mlu
python3 autotuner/tune_gemv.py --iterations 50 --repeat 1
```

Best configuration found so far:

```text
NRAM_CHUNK_FLOATS=1024
TILE_SRAM_BLOCK_ROWS=16
UNROLL_FACTOR=2
Best objective gmean_time_ms=0.292944
```

To compare the default `tile_sram_db` against this tuned configuration:

```bash
./autotuner/run_tile_sram_db_tuned_compare.sh \
  --repeat 3 \
  --out-dir build/perf_tile_sram_db_tuned
```

Tuned-vs-default result, repeated 3 times and averaged:

| Implementation | Pass | GMean Speedup | p50 | p90 | Min | Max |
|---|---:|---:|---:|---:|---:|---:|
| `tile_sram_db_default` | 33/33 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| `tile_sram_db_bo_best` | 33/33 | 1.1419 | 1.0714 | 1.4584 | 0.9000 | 1.8688 |

The tuned configuration improves the geometric mean by about `14.19%`. The strongest gains are on unit-stride cases (`1.2171x` gmean), while non-unit-stride cases improve less (`1.0671x` gmean).

To compare `tile_sram_xreuse` with both default and BO-tuned parameters:

```bash
./autotuner/run_tile_sram_xreuse_tuned_compare.sh \
  --repeat 3 \
  --out-dir build/perf_tile_sram_xreuse_tuned
```

This report uses `tile_sram_db_default` as the baseline and includes:

```text
tile_sram_db_default
tile_sram_xreuse_default
tile_sram_db_bo_best
tile_sram_xreuse_bo_best
```

Current x-reuse comparison result:

| Implementation | Pass | GMean Speedup | p50 | p90 | Min | Max |
|---|---:|---:|---:|---:|---:|---:|
| `tile_sram_db_default` | 33/33 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| `tile_sram_xreuse_default` | 33/33 | 1.1850 | 0.9902 | 2.5334 | 0.7068 | 2.7265 |
| `tile_sram_db_bo_best` | 33/33 | 1.1358 | 1.0707 | 1.4480 | 0.9000 | 1.8610 |
| `tile_sram_xreuse_bo_best` | 33/33 | 1.4007 | 1.0969 | 2.8743 | 0.9000 | 3.3696 |

`tile_sram_xreuse_bo_best` is the fastest configuration in this comparison. The benefit mainly comes from non-unit-stride cases: `incx=3,incy=2` reaches `1.6726x`, while `incx=1,incy=1` is `1.1854x`.

## Notes

- Use `analysis_avg/` reports for final comparisons rather than individual rounds.
- Use `GEMV_SEED=<seed>` to reproduce or vary test data.
- When changing compile-time kernel parameters, rerun CMake and rebuild.
- Current optimization work mainly targets A movement and reduction tiling; improving non-unit-stride x access is the next likely bottleneck.

## Decoder Decode Application Prototype

The repository now includes a phase-one application prototype that connects the single-row
Q/K/V/O and FFN projections of a minimal decoder layer to the existing GEMV API. The attention
softmax and KV-cache reference path are intentionally kept on the host for this first prototype;
all linear projections still go through the selected MLU GEMV implementation.

Build the application:

```bash
cmake -S . -B build \
  -DHAVE_MLU=ON \
  -DMLU_ARCH=270 \
  -DBUILD_APPS=ON
cmake --build build -j
```

Run a reproducible comparison between the scalar baseline and the current tiled kernel:

```bash
./build/bin/decoder_decode \
  --impl compare \
  --backend mlu \
  --hidden 256 \
  --intermediate 512 \
  --layers 1 \
  --tokens 1000 \
  --warmup 10 \
  --repeat 3 \
  --seed 20260925 \
  --report build/app/decoder_decode_report.md \
  --csv build/app/decoder_decode_report.csv
```

The report contains average end-to-end latency, milliseconds per token, GEMV/attention/other
time shares, relative speedup against the first backend, and one-token CPU-reference
verification. `--impl compare` compares `baseline` and `tile_sram_db`; `--impl all` compares
all seven MLU kernels; `--impl tile_sram_db` runs only the optimized path; `--impl baseline`
runs only the scalar path.

To compare all MLU kernel variants in the decoder workload:

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

The convenience script uses this all-kernel mode:

```bash
./autotuner/run_decoder_decode.sh
```

The `--impl all` result uses the compile-time parameters of the build that produced
`decoder_decode`. `GEMV_IMPL` only selects a kernel function; it does not switch the BO-tuned
parameters at runtime. To compare the default build against a BO configuration in the decoder
workload, use:

```bash
./autotuner/run_decoder_bo_compare.sh
```

This creates separate default and BO-tuned build directories and reports under:

```text
build/app_decoder_bo_compare/decoder_default.md
build/app_decoder_bo_compare/decoder_bo_best.md
```

The script defaults to the GEMV-microbenchmark BO configuration `1024/16/2`. For the current
decoder-specific BO result, override the parameters explicitly:

```bash
TUNED_NRAM=1024 \
TUNED_ROWS=16 \
TUNED_UNROLL=1 \
./autotuner/run_decoder_bo_compare.sh
```

If the seven-kernel decoder comparison has already been completed and only the BO result is
missing, run just the additional optimized `tile_sram_db` configuration:

```bash
./autotuner/run_decoder_bo_only.sh
```

This avoids rerunning the other six kernels and writes:

```text
build/app_decoder_bo_only/decoder_tile_sram_db_bo_best.md
build/app_decoder_bo_only/decoder_tile_sram_db_bo_best.csv
```

For the decoder-specific BO configuration:

```bash
BO_NRAM=1024 \
BO_ROWS=16 \
BO_UNROLL=1 \
./autotuner/run_decoder_bo_only.sh
```

To evaluate whether the best kernel changes with decoder dimensions, run the dimension sweep:

```bash
./autotuner/run_decoder_dimension_sweep.sh
```

It tests three configurations:

```text
small:  hidden=256,  intermediate=1024
medium: hidden=512,  intermediate=2048
large:  hidden=1024, intermediate=4096
```

Each configuration runs all seven default-parameter kernels plus one BO-parameter
`tile_sram_db` run. The default is `tokens=100`, `warmup=10`, `repeat=3`; override these with
environment variables when needed.

Current decoder-layer comparison results are shown below. The seven non-BO rows are from the
three-shape decoder sweep with default build parameters (`NRAM=256, ROWS=16, UNROLL=1`). Each
BO row uses the best parameter set found independently for that shape:

| Shape | NRAM | Rows | Unroll |
|---|---:|---:|---:|
| small (`256:1024`) | 512 | 16 | 4 |
| medium (`512:2048`) | 1024 | 16 | 4 |
| large (`1024:4096`) | 1024 | 8 | 4 |

All three tables below use the same final validation settings:
`tokens=1000`, `warmup=20`, `repeat=3`.

Small decoder shape, `hidden=256, intermediate=1024`:

| Backend | ms/token | Speedup | GEMV % | Attention % | Other % |
|---|---:|---:|---:|---:|---:|
| `mlu_baseline` | 52.362200 | 1.0000 | 89.04 | 10.58 | 0.38 |
| `mlu_tile_nram` | 9.416760 | 5.5605 | 53.11 | 45.43 | 1.45 |
| `mlu_tile_nram_db` | 9.011430 | 5.8106 | 54.81 | 43.95 | 1.24 |
| `mlu_tile_sram` | 8.903500 | 5.8811 | 54.39 | 44.35 | 1.26 |
| `mlu_tile_sram_db` | 8.846480 | 5.9190 | 54.47 | 44.28 | 1.25 |
| `mlu_tile_sram_xreuse` | 8.962620 | 5.8423 | 55.23 | 43.52 | 1.24 |
| `mlu_blas_style` | 8.960270 | 5.8438 | 54.99 | 43.76 | 1.25 |
| `mlu_tile_sram_db_bo_best` | 8.978730 | 5.8318 | 54.45 | 44.25 | 1.30 |

Medium decoder shape, `hidden=512, intermediate=2048`:

| Backend | ms/token | Speedup | GEMV % | Attention % | Other % |
|---|---:|---:|---:|---:|---:|
| `mlu_baseline` | 178.673000 | 1.0000 | 94.12 | 5.67 | 0.21 |
| `mlu_tile_nram` | 18.689100 | 9.5603 | 50.57 | 47.83 | 1.60 |
| `mlu_tile_nram_db` | 18.376600 | 9.7228 | 49.68 | 48.69 | 1.63 |
| `mlu_tile_sram` | 17.970400 | 9.9427 | 49.92 | 48.47 | 1.61 |
| `mlu_tile_sram_db` | 18.124000 | 9.8584 | 49.50 | 48.88 | 1.62 |
| `mlu_tile_sram_xreuse` | 16.821800 | 10.6215 | 52.00 | 46.67 | 1.33 |
| `mlu_blas_style` | 16.807100 | 10.6308 | 52.79 | 46.02 | 1.19 |
| `mlu_tile_sram_db_bo_best` | 16.747300 | 10.6688 | 52.31 | 46.41 | 1.28 |

Large decoder shape, `hidden=1024, intermediate=4096`:

| Backend | ms/token | Speedup | GEMV % | Attention % | Other % |
|---|---:|---:|---:|---:|---:|
| `mlu_baseline` | 627.040000 | 1.0000 | 97.11 | 2.78 | 0.11 |
| `mlu_tile_nram` | 42.576200 | 14.7275 | 59.74 | 38.95 | 1.31 |
| `mlu_tile_nram_db` | 40.172800 | 15.6086 | 57.92 | 40.72 | 1.36 |
| `mlu_tile_sram` | 40.217300 | 15.5913 | 57.61 | 41.02 | 1.37 |
| `mlu_tile_sram_db` | 39.955100 | 15.6936 | 57.81 | 40.83 | 1.36 |
| `mlu_tile_sram_xreuse` | 40.216700 | 15.5915 | 57.43 | 41.20 | 1.37 |
| `mlu_blas_style` | 39.953000 | 15.6945 | 57.76 | 40.88 | 1.36 |
| `mlu_tile_sram_db_bo_best` | 39.056400 | 16.0547 | 57.03 | 41.57 | 1.41 |

Across these decoder shapes, GEMV projection and attention are both important components in
the end-to-end prototype: optimized GEMV accounts for about `52%` to `58%`, while the CPU
reference attention path accounts for about `41%` to `47%`. The best decoder kernel is
shape-dependent, so decoder BO is reported per shape instead of using one shared parameter set
for all dimensions. In the final repeated validation, the per-shape BO improved the medium
shape by about `7.60%` and the large shape by about `2.25%` relative to the default
`tile_sram_db`; the small shape was about `1.49%` slower, which is within the fluctuation
observed between short BO evaluations and longer end-to-end runs.

To tune each decoder shape independently, run:

```bash
MPLCONFIGDIR=/tmp/mplconfig \
python3 autotuner/tune_decoder.py \
  --iterations 20 \
  --repeat 1 \
  --tokens 100 \
  --warmup 10 \
  --shapes 256:1024 \
  --build-dir build/bo_decoder_small \
  --out-dir build/bo_decoder_small

MPLCONFIGDIR=/tmp/mplconfig \
python3 autotuner/tune_decoder.py \
  --iterations 20 \
  --repeat 1 \
  --tokens 100 \
  --warmup 10 \
  --shapes 512:2048 \
  --build-dir build/bo_decoder_medium \
  --out-dir build/bo_decoder_medium

MPLCONFIGDIR=/tmp/mplconfig \
python3 autotuner/tune_decoder.py \
  --iterations 20 \
  --repeat 1 \
  --tokens 100 \
  --warmup 10 \
  --shapes 1024:4096 \
  --build-dir build/bo_decoder_large \
  --out-dir build/bo_decoder_large
```

After reading the best `NRAM/ROWS/UNROLL` from each `history.csv`, pass them to the sweep as
space-separated per-shape arrays:

```bash
python3 autotuner/summarize_decoder_bo.py \
  build/bo_decoder_small/history.csv \
  build/bo_decoder_medium/history.csv \
  build/bo_decoder_large/history.csv
```

```bash
BO_NRAM_VALUES="<small_nram> <medium_nram> <large_nram>" \
BO_ROWS_VALUES="<small_rows> <medium_rows> <large_rows>" \
BO_UNROLL_VALUES="<small_unroll> <medium_unroll> <large_unroll>" \
TOKENS=1000 \
WARMUP=20 \
REPEAT=3 \
OUT_DIR=build/final_decoder_compare \
./autotuner/run_decoder_dimension_sweep.sh
```

The original `tune_gemv.py` optimizes the 33-case GEMV microbenchmark. To tune
`tile_sram_db` directly for one decoder shape, use:

```bash
MPLCONFIGDIR=/tmp/mplconfig \
python3 autotuner/tune_decoder.py \
  --iterations 20 \
  --repeat 1 \
  --tokens 100 \
  --warmup 10 \
  --build-dir build/bo_decoder_safe \
  --out-dir build/bo_decoder_safe
```

With one shape in `--shapes`, the objective is that shape's `ms/token`. The history,
per-evaluation logs and convergence plot are written to the selected output directory. The
default decoder search space is:

```text
NRAM_CHUNK_FLOATS:     64, 128, 256, 512, 1024
TILE_SRAM_BLOCK_ROWS:  4, 8, 16
UNROLL_FACTOR:         1, 2, 4, 8
```

`TILE_SRAM_BLOCK_ROWS=32` is excluded from the default decoder search because it failed
decoder-layer correctness checks on the tested shapes. It can still be supplied manually with
`--rows-domain` if needed.

Per-shape BO search results with `tokens=100`, `warmup=10`, `repeat=1`:

| Shape | NRAM | Rows | Unroll | Best search ms/token |
|---|---:|---:|---:|---:|
| small (`256:1024`) | 512 | 16 | 4 | 5.293010 |
| medium (`512:2048`) | 1024 | 16 | 4 | 9.184070 |
| large (`1024:4096`) | 1024 | 8 | 4 | 22.859000 |

Final decoder validation uses `tokens=1000`, `warmup=20`, and `repeat=3`; those repeated
measurements are the values reported in the three tables above. The short BO search is used
to select candidates, while the repeated validation is used for the final performance claim.

The cnBLAS path is optional because this installation does not provide `cnblas.h` or
`libcnblas`. If a CNToolkit installation provides both, configure with:

```bash
cmake -S . -B build \
  -DHAVE_MLU=ON \
  -DMLU_ARCH=270 \
  -DBUILD_APPS=ON \
  -DGEMV_ENABLE_CNBLAS=ON
cmake --build build -j
./build/bin/decoder_decode --impl compare --backend all --repeat 3
```

The current phase-one prototype does not claim a true prefill GEMM result: this repository
contains GEMV kernels but no GEMM implementation. The report therefore measures the decode
path directly and records the prefill comparison as a limitation. A true prefill-vs-decode
comparison requires adding or linking a GEMM path in a follow-up step.
