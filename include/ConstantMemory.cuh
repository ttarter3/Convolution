// BLUR_SHARED_CONST_KERNEL_CUH
#ifndef CONSTANT_MEMORY_CUH
#define CONSTANT_MEMORY_CUH

// Cuda Headers
#include <math_constants.h>
#include <cuda_runtime.h>

#define CONSTMEMSIZE 2048
__constant__ float constant_filter[CONSTMEMSIZE];

#endif // CONSTANT_MEMORY_CUH