# GEMV

这是寒武纪的 GEMV 算子代码库，包含：

- GEMV Host API 与设备内存/句柄封装
- MLU GEMV kernel 实现
- GEMV 正确性与性能测试
- 基于贝叶斯优化的 GEMV 自适应调优脚本

## 目录结构

```text
.
├── CMakeLists.txt
├── include/
│   ├── GemvLibrary.h
│   ├── gemv.h
│   └── utils/
├── src/
│   ├── gemv.cpp
│   ├── kernels/mlu/gemv.mlu
│   └── utils/
├── tests/
│   └── test_gemv.cpp
└── autotuner/
    ├── tune_gemv.py
    └── launcher/gemv_bo_launcher.cpp
```

## 编译

默认启用 MLU 后端：

```bash
mkdir -p build
cd build
cmake -DHAVE_MLU=ON -DMLU_ARCH=270 ..
make -j
```

## 测试

`tests/test_gemv.cpp` 默认包含 33 个测试样例：

- 4 类矩阵规模：小型方阵、大型方阵、高瘦矩阵、宽扁矩阵
- 2 种转置模式：`N`、`T`
- 2 种 LDA：最小 LDA、大 LDA
- 2 种步长：单位步长、非单位步长
- 1 个边界样例：`1x1`

默认随机种子为 `20260613`，可通过 `GEMV_SEED` 指定，便于复现实验结果。

```bash
# 全部测试
./test/test_gemv

# 快速测试、列出样例、运行单个样例
./test/test_gemv quick
./test/test_gemv list
./test/test_gemv single 1

# CTest
ctest --output-on-failure
```

### 选择 kernel 实现

通过环境变量 `GEMV_IMPL` 选择要测试的 kernel：

```bash
cd /home/LCUDA/wjx/gemv_mlu/build

# baseline
GEMV_IMPL=baseline ./test/test_gemv

# 只做 GDRAM -> NRAM 分块搬运和 NRAM 向量计算
GEMV_IMPL=tile_nram ./test/test_gemv

# 在 tile_nram 基础上加入 NRAM ping-pong 双缓冲
GEMV_IMPL=tile_nram_db ./test/test_gemv

# 做 GDRAM -> SRAM -> NRAM 分块搬运，不使用双缓冲流水线
GEMV_IMPL=tile_sram ./test/test_gemv

# 在 tile_sram 基础上加入 SRAM ping-pong 双缓冲
GEMV_IMPL=tile_sram_db ./test/test_gemv

# BLAS 风格 prologue / steady-state / epilogue 流水实现
GEMV_IMPL=blas_style ./test/test_gemv

# 不设置 GEMV_IMPL 时也默认使用 blas_style
./test/test_gemv
```

常用单样例调试：

```bash
GEMV_IMPL=baseline ./test/test_gemv single 25
GEMV_IMPL=tile_nram ./test/test_gemv single 9
GEMV_IMPL=tile_nram_db ./test/test_gemv single 9
GEMV_IMPL=tile_sram ./test/test_gemv single 15
GEMV_IMPL=tile_sram_db ./test/test_gemv single 15
GEMV_IMPL=blas_style ./test/test_gemv single 9
```

如果需要更换测试数据：

```bash
GEMV_SEED=123 GEMV_IMPL=baseline ./test/test_gemv
```

## 性能对比

推荐用脚本一次性测试 `baseline -> tile_nram -> tile_nram_db -> tile_sram -> tile_sram_db -> blas_style` 六个阶段，并自动生成日志和加速比表格：

```bash
cd /home/LCUDA/wjx/gemv_mlu

./autotuner/run_gemv_perf_compare.sh \
  --repeat 3 \
  --out-dir build/perf_baseline_nram_sram
```

脚本会执行：

```text
GEMV_IMPL=baseline
GEMV_IMPL=tile_nram
GEMV_IMPL=tile_nram_db
GEMV_IMPL=tile_sram
GEMV_IMPL=tile_sram_db
GEMV_IMPL=blas_style
```

每轮都会生成独立日志和单轮分析结果；当 `--repeat > 1` 时，脚本还会额外生成三次平均后的最终分析结果：

```text
build/perf_baseline_nram_sram/
├── logs/
│   ├── baseline_r1.log
│   ├── tile_nram_r1.log
│   ├── tile_nram_db_r1.log
│   ├── tile_sram_r1.log
│   ├── tile_sram_db_r1.log
│   └── blas_style_r1.log
├── analysis_r1/
│   ├── gemv_speedup_summary.md
│   └── gemv_speedup_details.csv
├── analysis_r2/
├── analysis_r3/
└── analysis_avg/
    ├── gemv_speedup_summary.md
    └── gemv_speedup_details.csv
```

最终写性能表格时，优先使用 `analysis_avg/` 下的结果。`analysis_r1/analysis_r2/analysis_r3` 主要用于观察单轮波动。

快速检查可以只跑 quick：

```bash
./autotuner/run_gemv_perf_compare.sh --quick --repeat 3 --out-dir build/perf_quick
```

只对某个 case 做三实现对比：

```bash
./autotuner/run_gemv_perf_compare.sh --single 9 --out-dir build/perf_case9
```

### 结果解读

`gemv_speedup_summary.md` 是总览表，`baseline` 固定为 `1.0000`，其他实现的加速比计算方式为：

```text
speedup = baseline_time / impl_time
```

例如：

```text
baseline   gmean_speedup = 1.0000
tile_nram  gmean_speedup = 8.2936
tile_nram_db ...
tile_sram  gmean_speedup = 10.0467
tile_sram_db ...
blas_style ...
```

表示其他实现的耗时相对 baseline 的加速倍数。`tile_nram_db` 和 `tile_sram_db` 用于观察在 NRAM/SRAM 分块基础上加入双缓冲后的收益。

表格字段含义：

- `pass_count`：正确性通过数量。
- `gmean_speedup`：所有 case 加速比的几何平均，适合做总体性能结论。
- `p50`：加速比中位数。
- `p90`：较高分位加速比。
- `min`：最差 case 加速比，小于 1 表示比 baseline 慢。
- `max`：最好 case 加速比。

`gemv_speedup_details.csv` 是逐 case 细节。单轮分析中包含每个实现的耗时、带宽、正确性和相对 baseline 的加速比；平均分析中包含 `avg_time_ms`、`time_stdev_ms`、`avg_bandwidth_kbs`、`pass_count` 和基于平均耗时计算的加速比。写实验表格时建议优先使用 `analysis_avg/gemv_speedup_details.csv`。

### 双缓冲收益判断

双缓冲的收益需要结合 `tile_nram` 和 `tile_sram` 的基线分别判断，而不是只看相对 baseline 的加速比。推荐在 `gemv_speedup_details.csv` 中计算：

```text
tile_nram_db_vs_tile_nram = tile_nram_time / tile_nram_db_time
tile_sram_db_vs_tile_sram = tile_sram_time / tile_sram_db_time
```

当前测试中，`tile_nram_db` 对连续访存样例更有效，`incx=1` 时几何平均约有 `1.24x - 1.25x` 的额外收益；但 `incx=3` 时 x 需要逐元素 gather，双缓冲基本无法隐藏这种小粒度访存，收益接近 `1.0x`。`tile_sram_db` 的整体收益更小，通常只有约 `1.00x - 1.02x`，原因是 `tile_sram` 之后瓶颈不再只是 A 的 `GDRAM -> SRAM` 搬运，还包括 x 加载、`SRAM -> NRAM`、规约计算和每个 tile 后的同步开销。因此，GEMV 是 memory-bound 并不意味着双缓冲一定有明显收益，关键要看当前瓶颈是否正好是双缓冲能覆盖的那段搬运。

### 手动分析已有日志

如果已经有日志，也可以直接调用分析脚本：

```bash
cd /home/LCUDA/wjx/gemv_mlu

python3 autotuner/analyze_gemv_logs.py \
  --logs \
  baseline=build/results/baseline.log \
  tile_nram=build/results/tile_nram.log \
  tile_nram_db=build/results/tile_nram_db.log \
  tile_sram=build/results/tile_sram.log \
  tile_sram_db=build/results/tile_sram_db.log \
  blas_style=build/results/blas_style.log \
  --out-dir build/analysis
```

注意：`--logs` 的第一个输入会作为基准实现，因此一般把 `baseline=...` 放在第一个。

## 性能优化阶段说明

当前主要对比三个阶段：

- `baseline`：逐元素从 GDRAM 读入 NRAM，做标量乘加，作为性能基准。
- `tile_nram`：按 `NRAM_CHUNK_FLOATS` 分块，从 GDRAM 批量搬运到 NRAM，用 `__bang_mul` 和 `__bang_reduce_sum` 做向量化计算。
- `tile_nram_db`：在 `tile_nram` 基础上加入 NRAM ping-pong 双缓冲，当前 tile 计算时预取下一个 tile。
- `tile_sram`：按行块和列块分块，先将 A tile 从 GDRAM 搬到 SRAM，再由各 core 搬到 NRAM 计算，用于观察 `GDRAM -> SRAM -> NRAM` 数据路径的收益。
- `tile_sram_db`：在 `tile_sram` 基础上加入 SRAM ping-pong 双缓冲，当前 A tile 计算时预取下一个 A tile。
- `blas_style`：在 `tile_sram_db` 的 row block + x tile 复用基础上，将流水拆成 prologue / steady-state / epilogue 的 BLAS 风格实现。不设置 `GEMV_IMPL` 时也默认使用该 kernel。

分块带来加速的主要原因是：baseline 逐元素搬运会产生大量小粒度 GDRAM 访问，带宽利用率低；分块后可以批量连续搬运，并在 NRAM 上用向量指令完成乘法和规约。`tile_sram` 进一步改善了 A tile 的搬运和组织方式，因此通常会比 `tile_nram` 更快。对于 `incx != 1` 的样例，x 需要逐元素 gather，访存效率下降，加速幅度会明显小于单位步长样例。

当前数据表明，后续更值得优先优化的是非单位步长 `x` 的 gather、x tile 复用和同步开销，而不是继续单独增加 A tile 的双缓冲层级。

## 编译期参数

`src/kernels/mlu/gemv.mlu` 中保留了几个可调参数：

- `NRAM_CHUNK_FLOATS`：每次搬到 NRAM 的 float 数量，当前默认 `256`。
- `TILE_NRAM_BLOCK_ROWS`：`tile_nram` 的行分块大小，当前默认 `16`。
- `TILE_SRAM_BLOCK_ROWS`：`tile_sram` 的 SRAM 行分块大小，当前默认 `16`。
- `SRAM_BLOCK_ROWS`：旧默认优化版 kernel 的 SRAM 行分块大小，当前 `blas_style` 不使用。
- `PIPELINE_BUFFERS`：旧默认优化版 kernel 的流水缓冲数量，当前 `blas_style` 不使用。
- `UNROLL_FACTOR`：`blas_style` kernel 的尾部标量累加展开因子。

当前阶段建议先保持 `UNROLL_FACTOR=1`，也就是不做尾部循环展开，把 `blas_style` kernel 和前面分阶段实现先放在同一基准下比较。循环展开可以后续单独设置 `UNROLL_FACTOR=2/4` 再跑一组对比，避免把“BLAS 风格流水收益”和“循环展开收益”混在一起。

修改这些参数后需要重新编译：

```bash
cd /home/LCUDA/wjx/gemv_mlu/build
make -j
```

## 自适应调优

先完成编译，确保 `build/bin/gemv_bo_launcher` 存在，然后运行：

```bash
cd autotuner
python tune_gemv.py
```

调优脚本会搜索以下 kernel 编译期参数：

- `SRAM_BLOCK_ROWS`
- `NRAM_CHUNK_FLOATS`
- `PIPELINE_BUFFERS`
- `UNROLL_FACTOR`

并输出最佳参数和收敛曲线。
