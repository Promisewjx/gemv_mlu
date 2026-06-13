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

# 做 GDRAM -> SRAM -> NRAM 分块搬运，不使用双缓冲流水线
GEMV_IMPL=tile_sram ./test/test_gemv

# 不设置 GEMV_IMPL 时，使用默认优化版 kernel
./test/test_gemv
```

常用单样例调试：

```bash
GEMV_IMPL=baseline ./test/test_gemv single 25
GEMV_IMPL=tile_nram ./test/test_gemv single 9
GEMV_IMPL=tile_sram ./test/test_gemv single 15
```

如果需要更换测试数据：

```bash
GEMV_SEED=123 GEMV_IMPL=baseline ./test/test_gemv
```

## 性能对比

推荐用脚本一次性测试 `baseline -> tile_nram -> tile_sram` 三个阶段，并自动生成日志和加速比表格：

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
GEMV_IMPL=tile_sram
```

每轮都会生成独立日志和分析结果：

```text
build/perf_baseline_nram_sram/
├── logs/
│   ├── baseline_r1.log
│   ├── tile_nram_r1.log
│   └── tile_sram_r1.log
├── analysis_r1/
│   ├── gemv_speedup_summary.md
│   └── gemv_speedup_details.csv
├── analysis_r2/
└── analysis_r3/
```

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
tile_sram  gmean_speedup = 10.0467
```

表示在所有样例上，`tile_nram` 相比 baseline 的几何平均加速约为 `8.29x`，`tile_sram` 相比 baseline 的几何平均加速约为 `10.05x`。

表格字段含义：

- `pass_count`：正确性通过数量。
- `gmean_speedup`：所有 case 加速比的几何平均，适合做总体性能结论。
- `p50`：加速比中位数。
- `p90`：较高分位加速比。
- `min`：最差 case 加速比，小于 1 表示比 baseline 慢。
- `max`：最好 case 加速比。

`gemv_speedup_details.csv` 是逐 case 细节，包含每个实现的耗时、带宽、正确性和相对 baseline 的加速比。写实验表格时建议优先使用这个 CSV。

### 手动分析已有日志

如果已经有日志，也可以直接调用分析脚本：

```bash
cd /home/LCUDA/wjx/gemv_mlu

python3 autotuner/analyze_gemv_logs.py \
  --logs \
  baseline=build/results/baseline.log \
  tile_nram=build/results/tile_nram.log \
  tile_sram=build/results/tile_sram.log \
  --out-dir build/analysis
```

注意：`--logs` 的第一个输入会作为基准实现，因此一般把 `baseline=...` 放在第一个。

## 性能优化阶段说明

当前主要对比三个阶段：

- `baseline`：逐元素从 GDRAM 读入 NRAM，做标量乘加，作为性能基准。
- `tile_nram`：按 `NRAM_CHUNK_FLOATS` 分块，从 GDRAM 批量搬运到 NRAM，用 `__bang_mul` 和 `__bang_reduce_sum` 做向量化计算。
- `tile_sram`：按行块和列块分块，先将 A tile 从 GDRAM 搬到 SRAM，再由各 core 搬到 NRAM 计算，用于观察 `GDRAM -> SRAM -> NRAM` 数据路径的收益。

分块带来加速的主要原因是：baseline 逐元素搬运会产生大量小粒度 GDRAM 访问，带宽利用率低；分块后可以批量连续搬运，并在 NRAM 上用向量指令完成乘法和规约。`tile_sram` 进一步改善了 A tile 的搬运和组织方式，因此通常会比 `tile_nram` 更快。对于 `incx != 1` 的样例，x 需要逐元素 gather，访存效率下降，加速幅度会明显小于单位步长样例。

## 编译期参数

`src/kernels/mlu/gemv.mlu` 中保留了几个可调参数：

- `NRAM_CHUNK_FLOATS`：每次搬到 NRAM 的 float 数量，当前默认 `256`。
- `TILE_NRAM_BLOCK_ROWS`：`tile_nram` 的行分块大小，当前默认 `16`。
- `TILE_SRAM_BLOCK_ROWS`：`tile_sram` 的 SRAM 行分块大小，当前默认 `16`。
- `SRAM_BLOCK_ROWS`：默认优化版 kernel 的 SRAM 行分块大小，当前默认 `64`。
- `PIPELINE_BUFFERS`：默认优化版 kernel 的流水缓冲数量。
- `UNROLL_FACTOR`：默认优化版 kernel 的尾部标量累加展开因子。

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
