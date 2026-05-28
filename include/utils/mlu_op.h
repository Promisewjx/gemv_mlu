// 必须的 MLU 内建变量
#define taskDimX __task_dim.x
#define taskDimY __task_dim.y
#define taskIdX  __task_id.x
#define taskIdY  __task_id.y
#define coreDim  __core_dim
#define coreId   __core_id
#define taskDim  __task_dim.x
#define taskId   __task_id.x

#define __mlu_global__ __attribute__((global))
#define __mlu_func__   __attribute__((mlu_func))

// 内存方向（Cambricon 内部宏）
#define GDRAM2NRAM 0
#define SRAM2NRAM  1
#define NRAM2GDRAM 2
#define GDRAM2SRAM 3
#define SRAM2GDRAM 4
#define NRAM2SRAM  5

// 同步指令
#define __sync_cluster() __asm__ volatile("sync;")
