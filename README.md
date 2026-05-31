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
│   ├── test_gemv.cpp
│   └── test_gemv_union1.cpp
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

```bash
# 全部测试
./test/test_gemv

./test/test_gemv quick
./test/test_gemv list
./test/test_gemv single 1
./test/test_gemv_union1
ctest --output-on-failure
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
