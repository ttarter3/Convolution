// BLUR_CONSTANT_KERNEL_CUH
#ifndef BLUR_CONSTANT_KERNEL_CUH
#define BLUR_CONSTANT_KERNEL_CUH

#include "ConstantMemory.cuh"

// Cuda Headers
#include <math_constants.h>
#include <cuda_runtime.h>

template<typename T>
__global__ void BlurConstantKernel(T * img_in, T * img_out, int h_i, int w_i, T * filt_in, int w_f) {
    for(size_t col = blockIdx.y * blockDim.y + threadIdx.y; col < w_i; col += gridDim.y * blockDim.y) {
        for(size_t row = blockIdx.x * blockDim.x + threadIdx.x; row < h_i; row += gridDim.x * blockDim.x) {
            T p_value = 0.0f;
            for (int cc = 0; cc < w_f; cc++) {
                int in_col = col - (w_f + 1) / 2 + cc;
                for (int rr = 0; rr < w_f; rr++) {
                    int in_row = row - (w_f + 1) / 2 + rr;
                    if (in_row >= 0 && in_row < h_i && in_col >= 0 && in_col < w_i) {
                        p_value += constant_filter[rr * w_f + cc] * img_in[in_row * w_i + in_col];
                    }
                }
            }
            img_out[row * w_i + col] = p_value;
        }
    }
}
#endif // BLUR_CONSTANT_KERNEL_CUH