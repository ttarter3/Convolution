// BLUR_SHARED_EQUAL_CONST_KERNEL_CUH
#ifndef BLUR_SHARED_EQUAL_CONST_KERNEL_CUH
#define BLUR_SHARED_EQUAL_CONST_KERNEL_CUH

#include "ConstantMemory.cuh"

// Cuda Headers
#include <math_constants.h>
#include <cuda_runtime.h>



#define TILE_DIM 32
#define MAXFILT_DIM 11

template<typename T>
__global__ void BlurSharedEqualConstKernel(T * img_in, T * img_out, int h_i, int w_i, T * filt_in, int w_f) {
    int single_input_dim_size = TILE_DIM + (w_f + 1) / 2 * 2;
    __shared__ float tile[(TILE_DIM + MAXFILT_DIM) * (TILE_DIM + MAXFILT_DIM)];

    for(int col = blockIdx.y * TILE_DIM + threadIdx.y; col < w_i; col += gridDim.y * TILE_DIM) {
        for(int row = blockIdx.x * TILE_DIM + threadIdx.x; row < h_i; row += gridDim.x * TILE_DIM) {
            if (row >= 0 && row < h_i && col >= 0 && col < w_i) {
                tile[row * w_i + col] = img_in[row * w_i + col];
            } else {
                tile[row * w_i + col] = 0.0;
            }
            
            __syncthreads();

            if (row >= 0 && row < h_i && col >= 0 && col < w_i) {
                T p_value = 0.0f;
                for (int cc = 0; cc < w_f; cc++) {
                    for (int rr = 0; rr < w_f; rr++) {
                        if (
                           (int)threadIdx.x - (w_f + 1) / 2 + rr >= 0 
                        && (int)threadIdx.x - (w_f + 1) / 2 + rr < TILE_DIM
                        && (int)threadIdx.y - (w_f + 1) / 2 + cc >= 0 
                        && (int)threadIdx.y - (w_f + 1) / 2 + cc < TILE_DIM) {
                            p_value += constant_filter[rr * w_f + cc] * tile[(threadIdx.x + rr) * single_input_dim_size + (threadIdx.y + cc)];
                        } else if (
                                       row - (w_f + 1) / 2 + rr >= 0
                                    && row - (w_f + 1) / 2 + rr < h_i
                                    && col - (w_f + 1) / 2 + cc >= 0
                                    && col - (w_f + 1) / 2 + cc < w_i) {
                            p_value += constant_filter[rr * w_f + cc] * img_in[(row - (w_f + 1) / 2 + rr) * w_i + (col - (w_f + 1) / 2 + cc)];            

                        }
                        
                    }
                }
                img_out[row * w_i + col] = p_value;
            }
        }
    }
}
#endif // BLUR_SHARED_EQUAL_CONST_KERNEL_CUH