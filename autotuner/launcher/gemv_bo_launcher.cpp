#include <iostream>
#include <string>
#include <vector>
#include <cstdlib>
#include <memory>
#include <dlfcn.h>
#include <random>
#include <map>

#include <cnrt.h>
#include "GemvLibrary.h"

// 解析命令行参数
std::map<std::string, std::string> parse_args(int argc, char* argv[]) {
    std::map<std::string, std::string> args;
    for (int i = 1; i < argc; i += 2) {
        if (i + 1 < argc) {
            args[argv[i]] = argv[i + 1];
        }
    }
    return args;
}

// 初始化数据
void initialize_data(float* data, size_t size) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1.0, 1.0);
    for (size_t i = 0; i < size; ++i) {
        data[i] = dis(gen);
    }
}

// 定义函数指针类型
typedef void (*MluSgemvFunc)(DnHandle_t, char, int, int, const float, const float*, int,
                             const float*, int, const float, float*, int, void*);

int main(int argc, char* argv[]) {
    // === 1. 解析参数 ===
    auto args = parse_args(argc, argv);
    int sram_block_rows = args.count("--sram_rows") ? std::stoi(args["--sram_rows"]) : 64;
    int nram_chunk_floats = args.count("--nram_floats") ? std::stoi(args["--nram_floats"]) : 256;
    int pipeline_buffers = args.count("--pipeline_bufs") ? std::stoi(args["--pipeline_bufs"]) : 2;
    int unroll_factor = args.count("--unroll_factor") ? std::stoi(args["--unroll_factor"]) : 1;

    // === 2. 构建编译命令 ===
    std::string project_root = PROJECT_SOURCE_DIR;
    std::string kernel_path = project_root + "/src/kernels/mlu/gemv.mlu";
    std::string include_path = project_root + "/include";
    std::string output_so_path = "./gemv_kernel.so";

    std::string compile_cmd = std::string(CNCC_COMPILER_PATH) + " " + kernel_path + " -o " + output_so_path + " --shared";
    compile_cmd += " -fPIC";
    compile_cmd += " --bang-mlu-arch=mtp_270";
    compile_cmd += " -I" + include_path;
    compile_cmd += " -DSRAM_BLOCK_ROWS=" + std::to_string(sram_block_rows);
    compile_cmd += " -DNRAM_CHUNK_FLOATS=" + std::to_string(nram_chunk_floats);
    compile_cmd += " -DPIPELINE_BUFFERS=" + std::to_string(pipeline_buffers);
    compile_cmd += " -DUNROLL_FACTOR=" + std::to_string(unroll_factor);

    // === 3. 执行编译 ===
    int ret = system(compile_cmd.c_str());
    if (ret != 0) {
        std::cerr << "[ERROR] Kernel compilation failed." << std::endl;
        std::cout << "GFLOPS: 0.0" << std::endl;
        return 1;
    }

    // === 4. 动态加载库 ===
    void* handle = dlopen(output_so_path.c_str(), RTLD_LAZY);
    if (!handle) {
        std::cerr << "[ERROR] Failed to load shared library: " << dlerror() << std::endl;
        std::cout << "GFLOPS: 0.0" << std::endl;
        return 1;
    }
    MluSgemvFunc mlu_sgemv = (MluSgemvFunc)dlsym(handle, "mlu_BLAS_sGEMV");
    if (!mlu_sgemv) {
        std::cerr << "[ERROR] Failed to find symbol 'mlu_BLAS_sGEMV': " << dlerror() << std::endl;
        dlclose(handle);
        std::cout << "GFLOPS: 0.0" << std::endl;
        return 1;
    }

    // === 5. 初始化MLU设备和数据 ===
    // 底层初始化
    cnrtInit(0);
    cnrtQueue_t queue;
    cnrtCreateQueue(&queue);

    DnHandle_t dn_handle;
    dnCreate(&dn_handle);                 
    dnSetStream(dn_handle, (void*)queue);   

    // 准备数据
    int m = 16384, n = 16384;
    char trans = 'N';
    float alpha = 1.0f, beta = 0.0f;
    int lda = n, incx = 1, incy = 1;
    size_t a_size = (size_t)m * n, x_size = n, y_size = m;

    float *h_A = new float[a_size];
    float *h_x = new float[x_size];
    float *h_y = new float[y_size];

    initialize_data(h_A, a_size);
    initialize_data(h_x, x_size);
    initialize_data(h_y, y_size);

    float *d_A, *d_x, *d_y;
    cnrtMalloc((void**)&d_A, a_size * sizeof(float));
    cnrtMalloc((void**)&d_x, x_size * sizeof(float));
    cnrtMalloc((void**)&d_y, y_size * sizeof(float));

    cnrtMemcpy(d_A, h_A, a_size * sizeof(float), CNRT_MEM_TRANS_DIR_HOST2DEV);
    cnrtMemcpy(d_x, h_x, x_size * sizeof(float), CNRT_MEM_TRANS_DIR_HOST2DEV);
    cnrtMemcpy(d_y, h_y, y_size * sizeof(float), CNRT_MEM_TRANS_DIR_HOST2DEV);


    // === 6. 执行与计时 ===
    // 预热
    mlu_sgemv(dn_handle, trans, m, n, alpha, d_A, lda, d_x, incx, beta, d_y, incy, queue);

    // 正式计时
    const int iterations = 10;
    cnrtNotifier_t start, end;
    cnrtCreateNotifier(&start);
    cnrtCreateNotifier(&end);
    cnrtPlaceNotifier(start, queue);
    for (int i = 0; i < iterations; ++i) {
        mlu_sgemv(dn_handle, trans, m, n, alpha, d_A, lda, d_x, incx, beta, d_y, incy, queue);
    }
    cnrtPlaceNotifier(end, queue);
    cnrtSyncQueue(queue);

    float time_ms;
    cnrtNotifierDuration(start, end, &time_ms);
    float avg_time_ms = time_ms / iterations;

    // === 7. 计算并输出性能 ===
    double gflops = (2.0 * m * n) / (avg_time_ms / 1000.0) / 1e9;
    std::cout << "STATUS: SUCCESS | TIME_MS: " << avg_time_ms << " | GFLOPS: " << gflops << std::endl;

    // === 8. 清理 ===
    dlclose(handle);
    cnrtFree(d_A); cnrtFree(d_x); cnrtFree(d_y);
    delete[] h_A; delete[] h_x; delete[] h_y;
    dnDestroy(dn_handle); // 使用你的API销毁句柄
    cnrtDestroyQueue(queue);
    cnrtDestroyNotifier(&start);
    cnrtDestroyNotifier(&end);
    return 0;
}