#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ----------------------------
// 基本类型定义（来自 mlu_op.h）
// ----------------------------

typedef enum {
  MLUOP_STATUS_SUCCESS = 0,
  MLUOP_STATUS_BAD_PARAM = 1,
  MLUOP_STATUS_INTERNAL_ERROR = 2,
  MLUOP_STATUS_NOT_SUPPORTED = 3
} mluOpStatus_t;

// 数据类型枚举（如果你后续支持 half 可以加）
typedef enum {
  MLUOP_DTYPE_FLOAT = 0,
  MLUOP_DTYPE_HALF = 1
} mluOpDataType_t;

// 函数导出定义（Windows/非Windows 平台兼容）
#if defined(_WIN32) || defined(_WIN64)
#define MLUOP_WIN_API __declspec(dllexport)
#else
#define MLUOP_WIN_API
#endif

// ----------------------------
// 设备端调度结构体（来自 cnrt.h）
// ----------------------------

typedef struct {
  uint32_t x;
  uint32_t y;
  uint32_t z;
} cnrtDim3_t;

typedef enum {
  CNRT_FUNC_TYPE_BLOCK = 0,
  CNRT_FUNC_TYPE_UNION1 = 1,
  CNRT_FUNC_TYPE_UNION2 = 2,
  CNRT_FUNC_TYPE_UNION4 = 4,
  CNRT_FUNC_TYPE_UNION8 = 8
} cnrtFunctionType_t;

// 队列类型（本质是 void*）
typedef void* cnrtQueue_t;

#ifdef __cplusplus
}
#endif
