#include "GemvLibrary.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>

int main() {
  const int m = 4;
  const int n = 4;
  const float alpha = 1.0f;
  const float beta = 0.0f;

  // A: column-major, shape m x n
  float A[m * n] = {
      1, 5,  9, 13,
      2, 6, 10, 14,
      3, 7, 11, 15,
      4, 8, 12, 16
  };

  // x: vector of size n
  float x[n] = {1, 1, 1, 1};

  // y: vector of size m (output)
  float y[m] = {0, 0, 0, 0};

  // 创建 handle
  DnHandle_t handle;
  dnCreate(&handle);

  // 调用 GEMV
  DnStatus_t status = BLAS_sGEMV(
      handle, DnBLAS_OP_T, m, n, &alpha, A, m, x, 1, &beta, y, 1);

  if (status != DN_SUCCESS) {
    printf("GEMV failed with status %d\n", status);
    return -1;
  }

  // 打印输出 y
  printf("Result y = Aᵗ * x:\n");
  for (int i = 0; i < n; ++i) {
    printf("y[%d] = %.2f\n", i, y[i]);
  }

  dnDestroy(handle);
  return 0;
}
