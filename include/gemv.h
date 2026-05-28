#pragma once

#include "utils/handle.h"

#ifdef __cplusplus
extern "C" {
#endif

// BLAS Level2 GEMV: y = alpha * A * x + beta * y
// trans: Indicates the operation applied to matrix A
// m: Number of rows of matrix A
// n: Number of columns of matrix A
// alpha: scalar multiplier for A * x
// A: array of dimensions lda x n (column-major), with lda >= max(1, m)
// x: vector with n elements (DnOperation_N) or m elements (DnOperation_T/DnOperation_C)
// beta: scalar multiplier for vector y
// y: vector with m elements (DnOperation_N) or n elements (DnOperation_T/DnOperation_C)
DnStatus_t BLAS_sGEMV(DnHandle_t handle, DnOperation_t trans, int m, int n, const float *alpha,
                      const float *A, int lda, const float *x, int incx, const float *beta,
                      float *y, int incy);

#ifdef __cplusplus
} // extern "C"
#endif