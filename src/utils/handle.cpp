#include "utils/handle.h"
#include "utils/platform.h"
#include <cstdlib>
#include <cstdio>

#if defined(HAVE_MLU)
#include <cnrt.h>
#elif defined(HAVE_METAX)
#include <metax_runtime.h>
#endif

// Internal handle structure
struct DnHandle {
    int device_id;
    void *stream;
    bool owns_stream;
    float kernel_duration; 
#if defined(HAVE_MLU)
    cnrtQueue_t mlu_queue;
#elif defined(HAVE_METAX)
    metaxStream_t metax_stream;
#endif
};

DnStatus_t dnCreate(DnHandle_t *handle) {
    if (!handle)
        return DN_ERROR_INVALID_VALUE;

    *handle = (DnHandle_t)malloc(sizeof(struct DnHandle));
    if (!*handle)
        return DN_ERROR_ALLOC_FAILED;

    // Initialize with defaults
    // (*handle)->device_id = 3;
    (*handle)->device_id = 0;
    (*handle)->stream = nullptr;
    (*handle)->owns_stream = true;

#if defined(HAVE_MLU)
    // cnrtSetDevice((*handle)->device_id);
    cnrtInit(0);
    cnrtDev_t dev;
    cnrtGetDeviceHandle(&dev, (*handle)->device_id);
    cnrtSetCurrentDevice(dev);
    // if (cnrtQueueCreate(&((*handle)->mlu_queue)) != CNRT_RET_SUCCESS) {
    if (cnrtCreateQueue(&((*handle)->mlu_queue)) != CNRT_RET_SUCCESS) {
        free(*handle);
        return DN_ERROR_INTERNAL_ERROR;
    }
    (*handle)->stream = (*handle)->mlu_queue;
#elif defined(HAVE_METAX)
    metaxSetDevice((*handle)->device_id);
    if (metaxStreamCreate(&((*handle)->metax_stream)) != metaxSuccess) {
        free(*handle);
        return DN_ERROR_INTERNAL_ERROR;
    }
    (*handle)->stream = (*handle)->metax_stream;
#endif

    return DN_SUCCESS;
}

DnStatus_t dnDestroy(DnHandle_t handle) {
    if (!handle)
        return DN_ERROR_INVALID_VALUE;

#if defined(HAVE_MLU)
    if (handle->owns_stream && handle->mlu_queue) {
        // cnrtQueueDestroy(handle->mlu_queue);
        cnrtDestroyQueue(handle->mlu_queue);
    }
#elif defined(HAVE_METAX)
    if (handle->owns_stream && handle->metax_stream) {
        metaxStreamDestroy(handle->metax_stream);
    }
#endif

    free(handle);
    return DN_SUCCESS;
}

DnStatus_t dnSetDevice(DnHandle_t handle, int device) {
    if (!handle)
        return DN_ERROR_INVALID_VALUE;

    handle->device_id = device;

#if defined(HAVE_MLU)
    // if (cnrtSetDevice(device) != CNRT_RET_SUCCESS) {
    if (cnrtSetCurrentDevice(device) != CNRT_RET_SUCCESS) {
        return DN_ERROR_INTERNAL_ERROR;
    }
#elif defined(HAVE_METAX)
    if (metaxSetDevice(device) != metaxSuccess) {
        return DN_ERROR_INTERNAL_ERROR;
    }
#endif

    return DN_SUCCESS;
}

DnStatus_t dnGetDevice(DnHandle_t handle, int *device) {
    if (!handle || !device)
        return DN_ERROR_INVALID_VALUE;
    *device = handle->device_id;
    return DN_SUCCESS;
}

DnStatus_t dnSetStream(DnHandle_t handle, void *stream) {
    if (!handle)
        return DN_ERROR_INVALID_VALUE;

    // Destroy old stream if we own it
    if (handle->owns_stream) {
#if defined(HAVE_MLU)
        if (handle->mlu_queue)
            // cnrtQueueDestroy(handle->mlu_queue);
            cnrtDestroyQueue(handle->mlu_queue);
#elif defined(HAVE_METAX)
        if (handle->metax_stream)
            metaxStreamDestroy(handle->metax_stream);
#endif
    }

    handle->stream = stream;
    handle->owns_stream = false;

#if defined(HAVE_MLU)
    handle->mlu_queue = (cnrtQueue_t)stream;
#elif defined(HAVE_METAX)
    handle->metax_stream = (metaxStream_t)stream;
#endif

    return DN_SUCCESS;
}

DnStatus_t dnGetStream(DnHandle_t handle, void **stream) {
    if (!handle || !stream)
        return DN_ERROR_INVALID_VALUE;
    *stream = handle->stream;
    return DN_SUCCESS;
}

DnStatus_t dnSynchronize(DnHandle_t handle) {
    if (!handle)
        return DN_ERROR_INVALID_VALUE;

#if defined(HAVE_MLU)
    // if (cnrtQueueSync(handle->mlu_queue) != CNRT_RET_SUCCESS) {
    if (cnrtSyncQueue(handle->mlu_queue) != CNRT_RET_SUCCESS) {
        return DN_ERROR_EXECUTION_FAILED;
    }
#elif defined(HAVE_METAX)
    if (metaxStreamSynchronize(handle->metax_stream) != metaxSuccess) {
        return DN_ERROR_EXECUTION_FAILED;
    }
#endif

    return DN_SUCCESS;
}
DnStatus_t dnSetKernelDuration(DnHandle_t handle, float duration) {
    if (!handle)
        return DN_ERROR_INVALID_VALUE;
    handle->kernel_duration = duration;
    return DN_SUCCESS;
}

DnStatus_t dnGetKernelDuration(DnHandle_t handle, float *duration) {
    if (!handle || !duration)
        return DN_ERROR_INVALID_VALUE;
    *duration = handle->kernel_duration;
    return DN_SUCCESS;
}