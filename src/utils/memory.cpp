#include "utils/memory.h"
#include "utils/platform.h"

#if defined(HAVE_MLU)
#include <cnrt.h>
#elif defined(HAVE_METAX)
#include <metax_runtime.h>
#endif

DnStatus_t device_malloc(DnHandle_t handle, void **ptr, size_t size) {
    if (!handle || !ptr)
        return DN_ERROR_INVALID_VALUE;

#if defined(HAVE_MLU)
    if (cnrtMalloc(ptr, size) != CNRT_RET_SUCCESS) {
        return DN_ERROR_ALLOC_FAILED;
    }
#elif defined(HAVE_METAX)
    if (metaxMalloc(ptr, size) != metaxSuccess) {
        return DN_ERROR_ALLOC_FAILED;
    }
#endif
    return DN_SUCCESS;
}

DnStatus_t device_free(DnHandle_t handle, void *ptr) {
    if (!handle)
        return DN_ERROR_INVALID_VALUE;

#if defined(HAVE_MLU)
    if (cnrtFree(ptr) != CNRT_RET_SUCCESS) {
        return DN_ERROR_INTERNAL_ERROR;
    }
#elif defined(HAVE_METAX)
    if (metaxFree(ptr) != metaxSuccess) {
        return DN_ERROR_INTERNAL_ERROR;
    }
#endif
    return DN_SUCCESS;
}

DnStatus_t device_memset(DnHandle_t handle, void *ptr, int value, size_t size) {
    if (!handle || !ptr)
        return DN_ERROR_INVALID_VALUE;

#if defined(HAVE_MLU)
    if (cnrtMemset(ptr, value, size) != CNRT_RET_SUCCESS) {
        return DN_ERROR_ALLOC_FAILED;
    }
#elif defined(HAVE_METAX)
    if (metaxMemset(ptr, value, size) != metaxSuccess) {
        return DN_ERROR_ALLOC_FAILED;
    }
#endif
    return DN_SUCCESS;
}


DnStatus_t host_to_device_memcpy(DnHandle_t handle, void *dst, const void *src, size_t size) {
    if (!handle)
        return DN_ERROR_INVALID_VALUE;

    void *stream;
    dnGetStream(handle, &stream);

#if defined(HAVE_MLU)
    cnrtQueue_t queue = (cnrtQueue_t)stream;
    // if (cnrtMemcpyAsync(dst, const_cast<void *>(src), size, queue, cnrtMemcpyHostToDev) !=
    if (cnrtMemcpyAsync(dst, const_cast<void *>(src), size, queue, CNRT_MEM_TRANS_DIR_HOST2DEV) !=
        CNRT_RET_SUCCESS) {
        return DN_ERROR_EXECUTION_FAILED;
    }
    // if (cnrtMemcpy(dst, const_cast<void *>(src), size, CNRT_MEM_TRANS_DIR_HOST2DEV) != CNRT_RET_SUCCESS) {
    //     return DN_ERROR_EXECUTION_FAILED;
    // }
#elif defined(HAVE_METAX)
    metaxStream_t metax_stream = (metaxStream_t)stream;
    // if (metaxMemcpyAsync(dst, src, size, metaxMemcpyHostToDevice, metax_stream) != metaxSuccess) {
    //     return DN_ERROR_EXECUTION_FAILED;
    // }
    if (metaxMemcpy(dst, src, size, metaxMemcpyHostToDevice) != metaxSuccess) {
        return DN_ERROR_EXECUTION_FAILED;
    }
#endif
    return DN_SUCCESS;
}

DnStatus_t device_to_host_memcpy(DnHandle_t handle, void *dst, void *src, size_t size) {
    if (!handle)
        return DN_ERROR_INVALID_VALUE;

    void *stream;
    dnGetStream(handle, &stream);

#if defined(HAVE_MLU)
    cnrtQueue_t queue = (cnrtQueue_t)stream;
    // if (cnrtMemcpyAsync(dst, src, size, queue, cnrtMemcpyHostToDev) != CNRT_RET_SUCCESS) {
    if (cnrtMemcpyAsync(dst, src, size, queue, CNRT_MEM_TRANS_DIR_DEV2HOST) != CNRT_RET_SUCCESS) {
        return DN_ERROR_EXECUTION_FAILED;
    }
    // if (cnrtMemcpy(dst, src, size, CNRT_MEM_TRANS_DIR_DEV2HOST) != CNRT_RET_SUCCESS) {
    //     return DN_ERROR_EXECUTION_FAILED;
    // }
#elif defined(HAVE_METAX)
    // metaxStream_t metax_stream = (metaxStream_t)stream;
    // if (metaxMemcpyAsync(dst, src, size, metaxMemcpyDeviceToHost, metax_stream) != metaxSuccess) {
    //     return DN_ERROR_EXECUTION_FAILED;
    // }
    if (metaxMemcpy(dst, src, size, metaxMemcpyDeviceToHost) != metaxSuccess) {
        return DN_ERROR_EXECUTION_FAILED;
    }
#endif
    
    return DN_SUCCESS;
}
