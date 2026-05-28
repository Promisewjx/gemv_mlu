#include "GemvLibrary.h"
#include <cstdio>

#if defined(HAVE_MLU)
#include <cnrt.h>

#ifndef CNRT_CHECK
#define CNRT_CHECK(expr)                                                                               \
    do {                                                                                               \
        cnrtRet_t _ret = (expr);                                                                       \
        if (_ret != CNRT_RET_SUCCESS) {                                                                \
            std::fprintf(stderr, "CNRT error %d at %s:%d\\n", static_cast<int>(_ret), __FILE__,       \
                         __LINE__);                                                                    \
            return DN_ERROR_EXECUTION_FAILED;                                                          \
        }                                                                                              \
    } while (0)
#endif

// 声明 MLU kernel
extern "C" void mlu_BLAS_sGEMV(DnHandle_t handle,char trans, int m, int n, const float alpha, const float *d_A,
                               int lda, const float *d_x, int incx, const float beta, float *d_y,
                               int incy, void *stream);
// extern "C" void mlu_BLAS_sGEMV_union1(char trans, int m, int n, const float alpha, const float *d_A,
//                                int lda, const float *d_x, int incx, const float beta, float *d_y,
//                                int incy, void *stream);
#elif defined(HAVE_METAX)
#include <metax_runtime.h>
// 声明 MetaX kernel
extern "C" void metaX_BLAS_sGEMV(char trans, int m, int n, const float alpha, const float *d_A,
                                 int lda, const float *d_x, int incx, const float beta, float *d_y,
                                 int incy, void *stream);
#endif

// GEMV: y = alpha * A * x + beta * y
DnStatus_t BLAS_sGEMV(DnHandle_t handle, DnOperation_t trans, int m, int n, const float *alpha,
                      const float *A, int lda, const float *x, int incx, const float *beta,
                      float *y, int incy) {
    if (!handle || !A || !x || !y || !alpha || !beta || m <= 0 || n <= 0)
        return DN_ERROR_INVALID_VALUE;

    if (incx <= 0 || incy <= 0 || lda < m)
        return DN_ERROR_INVALID_INCREMENT;

    char c_trans;
    int x_len, y_len;

    // 将 DnOperation_t 映射到字符
    if (trans == DnBLAS_OP_N) {
        c_trans = 'N';
        x_len = n;
        y_len = m;
    } else if (trans == DnBLAS_OP_T) {
        c_trans = 'T';
        x_len = m;
        y_len = n;
    } else if (trans == DnBLAS_OP_C) {
        c_trans = 'C'; // 对于实数矩阵，共轭转置等同于转置
        x_len = m;
        y_len = n;
    } else {
        return DN_ERROR_INVALID_VALUE;
    }

    // 如果 alpha 为 0 且 beta 为 1，无需计算
    if (*alpha == 0.0f && *beta == 1.0f) {
        return DN_SUCCESS;
    }

    // 分配设备内存
    size_t x_size = sizeof(float) * ((x_len - 1) * incx + 1);
    size_t y_size = sizeof(float) * ((y_len - 1) * incy + 1);

    float *d_A = nullptr, *d_x = nullptr, *d_y = nullptr;
    DnStatus_t status;

    status = device_malloc(handle, (void **)&d_x, x_size);
    if (status != DN_SUCCESS)
        return status;

    status = device_malloc(handle, (void **)&d_y, y_size);
    if (status != DN_SUCCESS) {
        device_free(handle, d_x);
        return status;
    }

    float *A_T = nullptr; //  转换为行主存储后的矩阵
    bool need_transpose = (trans == DnBLAS_OP_N);

    if (need_transpose) {
        // 创建转置后的临时矩阵
        A_T = (float *)malloc(m * n * sizeof(float));
        if (!A_T) {
            device_free(handle, d_x);
            device_free(handle, d_y);
            return DN_ERROR_OUTOF_HOST_MEMORY;
        }

        // 执行矩阵列主序->行主序
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                A_T[i * n + j] = A[j * lda + i];
            }
        }

        // printf("Transposed matrix A:\n");
        // for (int i = 0; i < m; ++i) {
        //     for (int j = 0; j < n; ++j) {
        //         printf("%f ", A_T[i * n + j]);
        //     }
        //     printf("\n");
        // }
        // 打印转置后的矩阵
        // 确保转置后的矩阵大小正确

        // 分配转置后的设备内存（lda=n）
        status = device_malloc(handle, (void **)&d_A, m * n * sizeof(float));
        if (status != DN_SUCCESS) {
            free(A_T);
            device_free(handle, d_x);
            device_free(handle, d_y);
            return status;
        }

        status = host_to_device_memcpy(handle, d_A, A_T, m * n * sizeof(float));
    } else {
        // 矩阵本身已转置
        status = device_malloc(handle, (void **)&d_A, lda * n * sizeof(float));
        if (status != DN_SUCCESS) {
            device_free(handle, d_x);
            device_free(handle, d_y);
            return status;
        }

        // 拷贝原矩阵到设备
        status = host_to_device_memcpy(handle, d_A, A, lda * n * sizeof(float));
    }

    if (status != DN_SUCCESS) {
        device_free(handle, d_A);
        device_free(handle, d_x);
        device_free(handle, d_y);
        return status;
    }

    // 拷贝向量x和y到设备
    status = host_to_device_memcpy(handle, d_x, x, x_size);
    if (status != DN_SUCCESS) {
        dnSynchronize(handle);
        device_free(handle, d_A);
        device_free(handle, d_x);
        device_free(handle, d_y);
    }
    status = host_to_device_memcpy(handle, d_y, y, y_size);
    if (status != DN_SUCCESS) {
        dnSynchronize(handle);
        device_free(handle, d_A);
        device_free(handle, d_x);
        device_free(handle, d_y);
    }

    // 获取stream并启动设备kernel
    void *stream;
    dnGetStream(handle, &stream);
    int actual_lda = need_transpose ? n : lda;
    float duration = 0.0;

#if defined(HAVE_MLU)
    // mlu_BLAS_sGEMV(c_trans, m, n, *alpha, d_A, actual_lda, d_x, incx, *beta, d_y, incy, stream);
    // mlu_BLAS_sGEMV_union1(c_trans, m, n, *alpha, d_A, actual_lda, d_x, incx, *beta, d_y, incy, stream);
    cnrtNotifier_t start, end; // 创建计时器
    CNRT_CHECK(cnrtCreateNotifier(&start));
    CNRT_CHECK(cnrtCreateNotifier(&end));

    CNRT_CHECK(cnrtPlaceNotifier(start, (cnrtQueue_t)stream));
    mlu_BLAS_sGEMV(handle,c_trans, m, n, *alpha, d_A, actual_lda, d_x, incx, *beta, d_y, incy, stream);
    CNRT_CHECK(cnrtPlaceNotifier(end, (cnrtQueue_t)stream));

    cnrtSyncQueue((cnrtQueue_t)stream); // 等待设备执行完成

    CNRT_CHECK(cnrtNotifierDuration(start, end, &duration)); // 计算运行时间
    float duration_ms = duration / 1000.0f;
    dnSetKernelDuration(handle, duration_ms); // 设置内核执行时间
    CNRT_CHECK(cnrtDestroyNotifier(&start));
    CNRT_CHECK(cnrtDestroyNotifier(&end));
#elif defined(HAVE_METAX)
    metaX_BLAS_sGEMV(c_trans, m, n, *alpha, d_A, actual_lda, d_x, incx, *beta, d_y, incy, stream);
#endif

    // 拷贝结果回主机
    status = device_to_host_memcpy(handle, y, d_y, y_size);

    // 同步并释放设备内存
    dnSynchronize(handle);
    device_free(handle, d_A);
    device_free(handle, d_x);
    device_free(handle, d_y);
    if (need_transpose) free(A_T);

    return status;
}