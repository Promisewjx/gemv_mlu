#pragma once

#include "handle.h"
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// 设备内存管理接口
DnStatus_t device_malloc(DnHandle_t handle, void **ptr, size_t size);
DnStatus_t device_free(DnHandle_t handle, void *ptr);
DnStatus_t device_memset(DnHandle_t handle, void *ptr, int value, size_t size);
DnStatus_t host_to_device_memcpy(DnHandle_t handle, void *dst, const void *src, size_t size);
DnStatus_t device_to_host_memcpy(DnHandle_t handle, void *dst, void *src, size_t size);

#ifdef __cplusplus
} // extern "C"
#endif
