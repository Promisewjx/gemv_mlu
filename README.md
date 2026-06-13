# GEMV MLU

A Cambricon MLU implementation and benchmark suite for single-precision GEMV (`sgemv`). The project includes host-side wrappers, multiple BANG C kernel variants, correctness tests, performance comparison scripts, and a Bayesian autotuner for compile-time kernel parameters.

## Features

- MLU `sgemv` host API and utility wrappers.
- Multiple kernel implementations for baseline, NRAM tiling, SRAM tiling, double buffering, and BLAS-style pipelining.
- Correctness test suite covering 33 GEMV cases.
- Performance scripts that run repeated benchmarks and generate Markdown/CSV reports.
- Bayesian autotuning for the current best `tile_sram_db` kernel.

## Repository Layout

```text
.
├── include/                    # Public headers
├── src/
│   ├── gemv.cpp                # Host API entry
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
| `blas_style` | BLAS-style prologue / steady-state / epilogue implementation based on the SRAM tiled path. |

## Benchmark

Run the main benchmark comparison:

```bash
cd /home/LCUDA/wjx/gemv_mlu
./autotuner/run_gemv_perf_compare.sh --repeat 3 --out-dir build/perf
```

The script benchmarks:

```text
baseline -> tile_nram -> tile_nram_db -> tile_sram -> tile_sram_db -> blas_style
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

Main 33-case benchmark, repeated 3 times and averaged:

| Implementation | Pass | GMean Speedup | p50 | p90 | Min | Max |
|---|---:|---:|---:|---:|---:|---:|
| `baseline` | 33/33 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| `tile_nram` | 33/33 | 8.4158 | 5.0523 | 81.6053 | 0.7895 | 81.7812 |
| `tile_nram_db` | 33/33 | 9.3350 | 4.8202 | 145.4371 | 0.7143 | 154.0294 |
| `tile_sram` | 33/33 | 10.1134 | 5.6831 | 101.4254 | 0.6000 | 122.0473 |
| `tile_sram_db` | 33/33 | 10.1928 | 5.6344 | 110.3677 | 0.5556 | 129.8055 |
| `blas_style` | 33/33 | 9.6996 | 5.3419 | 105.1551 | 0.5000 | 129.0708 |

`tile_sram_db` is the best fixed implementation in this benchmark. Most of the gain comes from tiling and using the `GDRAM -> SRAM -> NRAM` path; the extra SRAM double buffer gives only a small additional gain over `tile_sram`. Non-unit-stride `x` remains the main weak point because it requires gather-style access.

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

## Notes

- Use `analysis_avg/` reports for final comparisons rather than individual rounds.
- Use `GEMV_SEED=<seed>` to reproduce or vary test data.
- When changing compile-time kernel parameters, rerun CMake and rebuild.
- Current optimization work mainly targets A movement and reduction tiling; improving non-unit-stride x access is the next likely bottleneck.
