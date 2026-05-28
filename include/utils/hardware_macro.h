#pragma once

// #if defined(HAVE_MLU)

#define MLU_CLUSTER_NUM (4)
#define MLU_CORE_DIM (4)
#define MLU_CORE_NUM (MLU_CLUSTER_NUM * MLU_CORE_DIM)

#define MAX_NRAM_SIZE (384 * 1024)   // 384KB, initialization value
#define MAX_SRAM_SIZE (1920 * 1024)  // 1920KB,initialization value

#define REM_FOR_STACK (128 * 1024)   // 128KB reserved for cncc

#define BINARY_NRAM_SIZE (MAX_NRAM_SIZE + REM_FOR_STACK - 112 * 1024)
#define BINARY_SRAM_SIZE (MLU_CORE_DIM * BINARY_NRAM_SIZE)

// threshold of bytes to be processed by each core
// according to the actual measurement results
#define MLU_ALIGN_SIZE 128

#define BINARY_ALIGN_NUM 64

// #endif

