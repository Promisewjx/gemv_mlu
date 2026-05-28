#include "GemvLibrary.h"
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <vector>

class GEMVTester {
  private:
    DnHandle_t handle;
    int test_count;
    int passed_count;
    std::vector<std::tuple<std::string, int, int, DnOperation_t, int, int, int>> all_tests;

    void initialize_matrix(std::vector<float> &matrix, int rows, int cols, int lda) {
        for (int j = 0; j < cols; j++) {
            for (int i = 0; i < rows; i++) {
                matrix[j * lda + i] = static_cast<float>(rand()) / RAND_MAX * 10.0f - 5.0f;
            }
        }
    }

    void initialize_vector(std::vector<float> &vector, int size, int inc) {
        for (int i = 0; i < size; i++) {
            vector[i * inc] = static_cast<float>(rand()) / RAND_MAX * 10.0f - 5.0f;
        }
    }

    void reference_gemv(DnOperation_t trans, int m, int n, float alpha, const std::vector<float> &A,
                    int lda, const std::vector<float> &x, int incx, float beta,
                    std::vector<float> &y, int incy) {
        bool is_trans = (trans != DnBLAS_OP_N);
        int x_len = is_trans ? m : n;
        int y_len = is_trans ? n : m;

        for (int i = 0; i < y_len; i++) {
            float sum = 0.0f;
            // if (i == 0 || i == 1) {
            //     std::cout << "[CPU] Row " << i << ": A_row[0..7] = ";
            //     for (int j = 0; j < std::min(8, x_len); ++j) {
            //         float a_val = is_trans ? A[i * lda + j] : A[j * lda + i];
            //         std::cout << a_val << " ";
            //     }
            //     std::cout << std::endl;
            //     std::cout << "[CPU] Row " << i << ": x[0..7] = ";
            //     for (int j = 0; j < std::min(8, x_len); ++j) {
            //         std::cout << x[j * incx] << " ";
            //     }
            //     std::cout << std::endl;
            // }
            for (int j = 0; j < x_len; j++) {
                float a_val;
                if (is_trans) {
                    a_val = A[i * lda + j];
                    // if (i < 3 && j < 10) {
                    //     std::cout << "[CPU] idx=" << i << ", j=" << j
                    //             << ", A=" << a_val << ", x=" << x[j * incx]
                    //             << ", sum=" << sum << std::endl;
                    // }
                } else {
                    a_val = A[j * lda + i];
                }
                sum += a_val * x[j * incx];
            }
            float old_y = y[i * incy];
            y[i * incy] = alpha * sum + beta * y[i * incy];
            // if (i == 0 || i == 1) {
            //     std::cout << "[CPU] idx=" << i << ", final sum=" << sum
            //             << ", y before=" << old_y
            //             << ", y after=" << y[i * incy] << std::endl;
            // }
        }
    }

    bool compare_vectors(const std::vector<float> &expected, const std::vector<float> &actual,
                         int size, int inc, float tolerance = 4e-2f) {
        for (int i = 0; i < size; i++) {
            float ref = expected[i * inc];
            float val = actual[i * inc];
            float denom = std::max(1.0f, std::fabs(ref)); // 防止除0
            float rel_diff = std::fabs(ref - val) / denom;
            if (rel_diff > tolerance) {
                std::cout << "    ❌ 错误: 索引 " << i << " 处不匹配: 期望值 " << ref
                        << ", 实际值 " << val << ", 相对误差=" << rel_diff << std::endl;
                return false;
            }
        }
        return true;
    }

    void run_test(const std::string &test_name, int m, int n, DnOperation_t trans, int lda_factor,
                  int incx, int incy) {
        test_count++;
        std::cout << "=== 测试 " << test_count << ": " << test_name << " ===" << std::endl;
        std::cout << "矩阵规模: " << m << "×" << n
                  << ", 转置: " << (trans == DnBLAS_OP_N ? "否" : "是")
                  << ", LDA倍数: " << lda_factor << ", incx: " << incx << ", incy: " << incy
                  << std::endl;

        int lda = lda_factor * m;
        bool is_trans = (trans != DnBLAS_OP_N);
        int x_len = is_trans ? m : n;
        int y_len = is_trans ? n : m;

        // Initialize data
        std::vector<float> A(lda * n, 0.0f);
        std::vector<float> x((x_len - 1) * incx + 1, 0.0f);
        std::vector<float> y((y_len - 1) * incy + 1, 0.0f);
        std::vector<float> y_expected = y;

        initialize_matrix(A, m, n, lda);
        initialize_vector(x, x_len, incx);
        initialize_vector(y, y_len, incy);
        y_expected = y;

        float alpha = 1.5f;
        float beta = 0.5f;

        // Reference computation
        reference_gemv(trans, m, n, alpha, A, lda, x, incx, beta, y_expected, incy);

        // kernel 调用前
        // std::cout << "[DEBUG] y y_expected kernel: ";
        // for (int i = 0; i < std::min(3, y_len); ++i) std::cout << y_expected[i * incy] << " ";
        // std::cout << std::endl;

        // for (int idx = 0; idx < 3; ++idx) {
        //     std::cout << "[HOST] idx=" << idx
        //             << ", A[0][idx]=" << A[0 * lda + idx]
        //             << ", A[1][idx]=" << A[1 * lda + idx]
        //             << ", A[2][idx]=" << A[2 * lda + idx] << std::endl;
        // }

        // GEMV library computation
        DnStatus_t status = BLAS_sGEMV(handle, trans, m, n, &alpha, A.data(), lda, x.data(), incx,
                                    &beta, y.data(), incy);
        // kernel 调用后
        // std::cout << "[DEBUG] y after kernel: ";
        // for (int i = 0; i < std::min(3, y_len); ++i) std::cout << y[i * incy] << " ";
        // std::cout << std::endl;

        // 获取kernel耗时
        float kernel_ms = 0.0f;
        dnGetKernelDuration(handle, &kernel_ms);
        
        size_t bytes = m * n * sizeof(float) + x_len * sizeof(float) + 2 * y_len * sizeof(float);
        float seconds = kernel_ms / 1000.0f;
        float gb = bytes / 1e9f;
        float bandwidth = gb / seconds;
        float bandwidth_kb = bandwidth * 1e6f;
        std::cout << "Kernel耗时: " << kernel_ms << " ms, 带宽: "
                << std::fixed << std::setprecision(3) << bandwidth_kb << " KB/s" << std::endl;

        if (status != DN_SUCCESS) {
            std::cout << "❌ 测试失败: BLAS_sGEMV 返回错误代码 " << status << std::endl;
            std::cout << std::endl;
            return;
        }

        // Compare results
        if (compare_vectors(y_expected, y, y_len, incy)) {
            std::cout << "✅ " << test_name << " 测试通过" << std::endl;
            passed_count++;
        } else {
            std::cout << "❌ " << test_name << " 测试失败: 结果不匹配" << std::endl;
        }
        std::cout << std::endl;
    }

    void setup_all_tests() {
        all_tests.clear();

        // Test matrix sizes - reduced for better accuracy
        std::vector<std::tuple<int, int, std::string>> sizes = {{25, 25, "小型方阵"},
                                                                {2000, 2000, "大型方阵"},
                                                                {10000, 20, "高瘦矩阵"},
                                                                {20, 10000, "宽扁矩阵"}};

        // Generate all combinations: 4 sizes × 2 transpose × 2 lda × 2 increment = 32 tests
        for (const auto &size_info : sizes) {
            int m = std::get<0>(size_info);
            int n = std::get<1>(size_info);
            std::string size_name = std::get<2>(size_info);

            for (int trans_idx = 0; trans_idx < 2; trans_idx++) {
                DnOperation_t trans = (trans_idx == 0) ? DnBLAS_OP_N : DnBLAS_OP_T;
                std::string trans_name = (trans_idx == 0) ? "无转置" : "转置";

                for (int lda_idx = 0; lda_idx < 2; lda_idx++) {
                    int lda_factor = (lda_idx == 0) ? 1 : 20;
                    std::string lda_name = (lda_idx == 0) ? "最小LDA" : "大LDA";

                    for (int inc_idx = 0; inc_idx < 2; inc_idx++) {
                        int incx = (inc_idx == 0) ? 1 : 3;
                        int incy = (inc_idx == 0) ? 1 : 2;
                        std::string inc_name = (inc_idx == 0) ? "单位步长" : "非单位步长";

                        std::string test_name =
                            size_name + "_" + trans_name + "_" + lda_name + "_" + inc_name;
                        all_tests.push_back(
                            std::make_tuple(test_name, m, n, trans, lda_factor, incx, incy));
                    }
                }
            }
        }

        // Add boundary test (33rd test)
        all_tests.push_back(std::make_tuple("边界_1x1矩阵", 1, 1, DnBLAS_OP_N, 1, 1, 1));
    }

  public:
    GEMVTester() : test_count(0), passed_count(0) {
        DnStatus_t status = dnCreate(&handle);
        if (status != DN_SUCCESS) {
            std::cerr << "❌ 创建 DnHandle 失败" << std::endl;
            exit(1);
        }
        srand(time(nullptr));
        setup_all_tests();
    }

    ~GEMVTester() { dnDestroy(handle); }

    void run_all_tests() {
        std::cout << "🚀 === GEMV 综合测试套件 ===" << std::endl;
        std::cout << "即将运行 " << all_tests.size() << " 个测试用例..." << std::endl << std::endl;

        for (const auto &test_info : all_tests) {
            std::string name = std::get<0>(test_info);
            int m = std::get<1>(test_info);
            int n = std::get<2>(test_info);
            DnOperation_t trans = std::get<3>(test_info);
            int lda_factor = std::get<4>(test_info);
            int incx = std::get<5>(test_info);
            int incy = std::get<6>(test_info);

            run_test(name, m, n, trans, lda_factor, incx, incy);
        }

        // Summary
        std::cout << "🎯 === 测试总结 ===" << std::endl;
        std::cout << "总测试数: " << test_count << std::endl;
        std::cout << "通过测试: " << passed_count << std::endl;
        std::cout << "失败测试: " << (test_count - passed_count) << std::endl;

        float success_rate = 100.0 * passed_count / test_count;
        std::cout << "成功率: " << std::fixed << std::setprecision(2) << success_rate << "%"
                  << std::endl;

        if (passed_count == test_count) {
            std::cout << "🎉 所有测试通过！" << std::endl;
        } else {
            std::cout << "⚠️  存在失败的测试，请检查上述输出。" << std::endl;
        }
    }

    void run_quick_test() {
        std::cout << "⚡ === GEMV 快速测试 ===" << std::endl;

        // Run a few representative tests
        run_test("快速_小型_无转置_最小LDA_单位步长", 25, 25, DnBLAS_OP_N, 1, 1, 1);
        run_test("快速_小型_转置_最小LDA_单位步长", 25, 25, DnBLAS_OP_T, 1, 1, 1);
        run_test("快速_高瘦_无转置_最小LDA_非单位步长", 1000, 10, DnBLAS_OP_N, 1, 2, 3);
        run_test("快速_宽扁_转置_大LDA_单位步长", 10, 1000, DnBLAS_OP_T, 5, 1, 1);
        // run_test("调试_3x3_无转置", 3, 3, DnBLAS_OP_N, 1, 1, 1);

        std::cout << "🎯 === 快速测试总结 ===" << std::endl;
        std::cout << "通过: " << passed_count << "/" << test_count << std::endl;

        if (passed_count == test_count) {
            std::cout << "✅ 快速测试全部通过！" << std::endl;
        } else {
            std::cout << "❌ 快速测试存在失败案例。" << std::endl;
        }
    }

    void list_all_tests() {
        std::cout << "📋 === 可用测试列表 ===" << std::endl;
        std::cout << "总共 " << all_tests.size() << " 个测试用例:" << std::endl << std::endl;

        for (size_t i = 0; i < all_tests.size(); i++) {
            const auto &test_info = all_tests[i];
            std::string name = std::get<0>(test_info);
            int m = std::get<1>(test_info);
            int n = std::get<2>(test_info);
            DnOperation_t trans = std::get<3>(test_info);
            int lda_factor = std::get<4>(test_info);
            int incx = std::get<5>(test_info);
            int incy = std::get<6>(test_info);

            std::cout << std::setw(2) << (i + 1) << ". " << name << std::endl;
            std::cout << "    规模: " << m << "×" << n
                      << ", 转置: " << (trans == DnBLAS_OP_N ? "否" : "是")
                      << ", LDA倍数: " << lda_factor << ", incx: " << incx << ", incy: " << incy
                      << std::endl;
            std::cout << std::endl;
        }

        std::cout << "💡 使用方法:" << std::endl;
        std::cout << "  运行单个测试: ./test_gemv single <测试编号>" << std::endl;
        std::cout << "  运行单个测试: ./test_gemv single <测试名称关键词>" << std::endl;
        std::cout << "  示例: ./test_gemv single 1" << std::endl;
        std::cout << "  示例: ./test_gemv single 小型方阵_无转置" << std::endl;
    }

    void run_single_test_by_index(int index) {
        if (index < 1 || index > static_cast<int>(all_tests.size())) {
            std::cout << "❌ 无效的测试编号: " << index << std::endl;
            std::cout << "有效范围: 1-" << all_tests.size() << std::endl;
            return;
        }

        const auto &test_info = all_tests[index - 1];
        std::string name = std::get<0>(test_info);
        int m = std::get<1>(test_info);
        int n = std::get<2>(test_info);
        DnOperation_t trans = std::get<3>(test_info);
        int lda_factor = std::get<4>(test_info);
        int incx = std::get<5>(test_info);
        int incy = std::get<6>(test_info);

        std::cout << "🎯 === 运行单个测试 ===" << std::endl;
        std::cout << "测试编号: " << index << std::endl;
        run_test(name, m, n, trans, lda_factor, incx, incy);

        std::cout << "📊 === 单测结果 ===" << std::endl;
        if (passed_count > 0) {
            std::cout << "✅ 测试通过" << std::endl;
        } else {
            std::cout << "❌ 测试失败" << std::endl;
        }
    }

    void run_single_test_by_name(const std::string &keyword) {
        std::vector<int> matching_indices;

        // Find tests that contain the keyword
        for (size_t i = 0; i < all_tests.size(); i++) {
            const std::string &test_name = std::get<0>(all_tests[i]);
            if (test_name.find(keyword) != std::string::npos) {
                matching_indices.push_back(i + 1);
            }
        }

        if (matching_indices.empty()) {
            std::cout << "❌ 未找到包含关键词 '" << keyword << "' 的测试" << std::endl;
            std::cout << "💡 提示: 使用 './test_gemv list' 查看所有可用测试" << std::endl;
            return;
        }

        if (matching_indices.size() == 1) {
            // Exact match, run the test
            run_single_test_by_index(matching_indices[0]);
        } else {
            // Multiple matches, let user choose
            std::cout << "🔍 找到 " << matching_indices.size() << " 个匹配的测试:" << std::endl;
            for (int idx : matching_indices) {
                const std::string &test_name = std::get<0>(all_tests[idx - 1]);
                std::cout << "  " << idx << ". " << test_name << std::endl;
            }
            std::cout << "请使用具体的测试编号运行测试" << std::endl;
        }
    }
};

// ./test_gemv [mode] [args...]
// mode: "quick" - 快速测试, "list" - 列出所有测试, "single <编号|关键词>" - 运行单个测试
// args: 仅在 mode 为 "single" 时需要，指定测试编号或关键词
// 如果没有参数，则运行所有测试

int main(int argc, char *argv[]) {
    GEMVTester tester;

    if (argc == 1) {
        // Default: run all tests
        tester.run_all_tests();
    } else if (argc == 2) {
        std::string mode = argv[1];
        if (mode == "quick") {
            tester.run_quick_test();
        } else if (mode == "list") {
            tester.list_all_tests();
        } else {
            std::cout << "❌ 未知参数: " << mode << std::endl;
            std::cout << "💡 用法:" << std::endl;
            std::cout << "  ./test_gemv          - 运行所有测试" << std::endl;
            std::cout << "  ./test_gemv quick    - 运行快速测试" << std::endl;
            std::cout << "  ./test_gemv list     - 列出所有测试" << std::endl;
            std::cout << "  ./test_gemv single <编号|关键词> - 运行单个测试" << std::endl;
        }
    } else if (argc == 3) {
        std::string mode = argv[1];
        if (mode == "single") {
            std::string target = argv[2];

            // Try to parse as number first
            try {
                int index = std::stoi(target);
                tester.run_single_test_by_index(index);
            } catch (const std::exception &) {
                // Not a number, treat as keyword
                tester.run_single_test_by_name(target);
            }
        } else {
            std::cout << "❌ 未知参数组合" << std::endl;
        }
    } else {
        std::cout << "❌ 参数过多" << std::endl;
        std::cout << "💡 用法:" << std::endl;
        std::cout << "  ./test_gemv          - 运行所有测试" << std::endl;
        std::cout << "  ./test_gemv quick    - 运行快速测试" << std::endl;
        std::cout << "  ./test_gemv list     - 列出所有测试" << std::endl;
        std::cout << "  ./test_gemv single <编号|关键词> - 运行单个测试" << std::endl;
    }

    return 0;
}

// int main() {
//     DnHandle_t handle;
//     if (dnCreate(&handle) != DN_SUCCESS) {
//         std::cerr << "❌ 创建 DnHandle 失败" << std::endl;
//         return 1;
//     }

//     // 简单3x3 GEMV
//     int m = 3, n = 3, lda = 3, incx = 1, incy = 1;
//     float alpha = 1.0f, beta = 0.0f;
//     DnOperation_t trans = DnBLAS_OP_N;

//     // A = [1 2 3; 4 5 6; 7 8 9]
//     std::vector<float> A = {
//         1, 4, 7,
//         2, 5, 8,
//         3, 6, 9
//     };
//     // x = [1, 2, 3]
//     std::vector<float> x = {1, 2, 3};
//     // y = [0, 0, 0]
//     std::vector<float> y = {0, 0, 0};
//     std::vector<float> y_expected = y;

//     // 参考实现
//     for (int i = 0; i < m; ++i) {
//         float sum = 0.0f;
//         for (int j = 0; j < n; ++j) {
//             sum += A[j * lda + i] * x[j * incx];
//         }
//         y_expected[i * incy] = alpha * sum + beta * y[i * incy];
//     }

//     // 打印输入
//     std::cout << "A:" << std::endl;
//     for (int i = 0; i < m; ++i) {
//         for (int j = 0; j < n; ++j) {
//             std::cout << A[j * lda + i] << " ";
//         }
//         std::cout << std::endl;
//     }
//     std::cout << "x: ";
//     for (auto v : x) std::cout << v << " ";
//     std::cout << std::endl;

//     // 调用 MLU kernel
//     DnStatus_t status = BLAS_sGEMV(handle, trans, m, n, &alpha, A.data(), lda, x.data(), incx, &beta, y.data(), incy);

//     // 打印输出
//     std::cout << "y_expected: ";
//     for (auto v : y_expected) std::cout << v << " ";
//     std::cout << std::endl;
//     std::cout << "y (mlu):    ";
//     for (auto v : y) std::cout << v << " ";
//     std::cout << std::endl;

//     if (status != DN_SUCCESS) {
//         std::cout << "❌ BLAS_sGEMV 返回错误代码 " << status << std::endl;
//     } else {
//         bool ok = true;
//         for (int i = 0; i < m; ++i) {
//             if (std::fabs(y[i] - y_expected[i]) > 1e-3) ok = false;
//         }
//         if (ok) {
//             std::cout << "✅ 简单GEMV测试通过" << std::endl;
//         } else {
//             std::cout << "❌ 简单GEMV测试失败" << std::endl;
//         }
//     }

//     dnDestroy(handle);
//     return 0;
// }

// int main() {
//     DnHandle_t handle;
//     if (dnCreate(&handle) != DN_SUCCESS) {
//         std::cerr << "❌ 创建 DnHandle 失败" << std::endl;
//         return 1;
//     }

//     // 简单4x4 GEMV
//     int m = 4, n = 4, lda = 4, incx = 1, incy = 1;
//     float alpha = 1.0f, beta = 0.0f;
//     DnOperation_t trans = DnBLAS_OP_N;

//     // A = [1  2  3  4;
//     //      5  6  7  8;
//     //      9 10 11 12;
//     //     13 14 15 16]
//     std::vector<float> A = {
//         1, 5, 9, 13,
//         2, 6,10, 14,
//         3, 7,11, 15,
//         4, 8,12, 16
//     };
//     // x = [1, 2, 3, 4]
//     std::vector<float> x = {1, 2, 3, 4};
//     // y = [0, 0, 0, 0]
//     std::vector<float> y = {0, 0, 0, 0};
//     std::vector<float> y_expected = y;

//     // 参考实现
//     for (int i = 0; i < m; ++i) {
//         float sum = 0.0f;
//         for (int j = 0; j < n; ++j) {
//             sum += A[j * lda + i] * x[j * incx];
//         }
//         y_expected[i * incy] = alpha * sum + beta * y[i * incy];
//     }

//     // 打印输入
//     std::cout << "A:" << std::endl;
//     for (int i = 0; i < m; ++i) {
//         for (int j = 0; j < n; ++j) {
//             std::cout << A[j * lda + i] << " ";
//         }
//         std::cout << std::endl;
//     }
//     std::cout << "x: ";
//     for (auto v : x) std::cout << v << " ";
//     std::cout << std::endl;

//     // 调用 MLU kernel
//     DnStatus_t status = BLAS_sGEMV(handle, trans, m, n, &alpha, A.data(), lda, x.data(), incx, &beta, y.data(), incy);

//     // 打印输出
//     std::cout << "y_expected: ";
//     for (auto v : y_expected) std::cout << v << " ";
//     std::cout << std::endl;
//     std::cout << "y (mlu):    ";
//     for (auto v : y) std::cout << v << " ";
//     std::cout << std::endl;

//     if (status != DN_SUCCESS) {
//         std::cout << "❌ BLAS_sGEMV 返回错误代码 " << status << std::endl;
//     } else {
//         bool ok = true;
//         for (int i = 0; i < m; ++i) {
//             if (std::fabs(y[i] - y_expected[i]) > 1e-3) ok = false;
//         }
//         if (ok) {
//             std::cout << "✅ 简单GEMV测试通过" << std::endl;
//         } else {
//             std::cout << "❌ 简单GEMV测试失败" << std::endl;
//         }
//     }

//     dnDestroy(handle);
//     return 0;
// }
