#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// Handle structure (opaque to users)
typedef struct DnHandle *DnHandle_t;

// Handle management functions
typedef enum {
    DN_SUCCESS = 0,
    DN_ERROR_NOT_INITIALIZED,
    DN_ERROR_ALLOC_FAILED,
    DN_ERROR_INVALID_VALUE,
    DN_ERROR_ARCH_MISMATCH,
    DN_ERROR_MAPPING_ERROR,
    DN_ERROR_EXECUTION_FAILED,
    DN_ERROR_INTERNAL_ERROR,
    DN_ERROR_NOT_SUPPORTED,
    DN_ERROR_INVALID_INCREMENT,
    DN_ERROR_OUTOF_HOST_MEMORY,
} DnStatus_t;

typedef enum {
    CUBLAS_FILL_MODE_LOWER,
    CUBLAS_FILL_MODE_UPPER
} DnFillMode_t;

typedef enum { 
    DnBLAS_OP_N, 
    DnBLAS_OP_T, 
    DnBLAS_OP_C 
} DnOperation_t;

typedef enum {
    DnBLAS_DIAG_NON_UNIT,
    DnBLAS_DIAG_UNIT
} DnDiagType_t;

typedef enum {
    DnBLAS_SIDE_LEFT,  
    DnBLAS_SIDE_RIGHT 
} DnSideMode_t;


// Handle lifecycle
DnStatus_t dnCreate(DnHandle_t *handle);
DnStatus_t dnDestroy(DnHandle_t handle);

// Device management
DnStatus_t dnSetDevice(DnHandle_t handle, int device);
DnStatus_t dnGetDevice(DnHandle_t handle, int *device);

// Stream/Queue management
DnStatus_t dnSetStream(DnHandle_t handle, void *stream);
DnStatus_t dnGetStream(DnHandle_t handle, void **stream);

// Synchronization
DnStatus_t dnSynchronize(DnHandle_t handle);

DnStatus_t dnSetKernelDuration(DnHandle_t handle, float duration);
DnStatus_t dnGetKernelDuration(DnHandle_t handle, float *duration);

#ifdef __cplusplus
} // extern "C"
#endif
