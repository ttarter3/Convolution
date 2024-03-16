// BLUR_SHARED_SHARED_KERNEL_CUH
#ifndef BLUR_SHARED_SHARED_KERNEL_CUH
#define BLUR_SHARED_SHARED_KERNEL_CUH

// Cuda Headers
#include <math_constants.h>
#include <cuda_runtime.h>

#define TILE_DIM 32
#define MAXFILT_DIM 11
#define SHRMEMSIZE 2048

template<typename T>
__global__ void BlurSharedSharedKernel(T * img_in, T * img_out, int h_i, int w_i, T * filt_in, int w_f) {
    __shared__ float shared_filter[SHRMEMSIZE];

    int single_input_dim_size = TILE_DIM + (w_f + 1) / 2 * 2;
    __shared__ float tile[(TILE_DIM + MAXFILT_DIM) * (TILE_DIM + MAXFILT_DIM)];

    for(int col = blockIdx.y * blockDim.y + threadIdx.y - (w_f + 1) / 2; col < w_i; col += gridDim.y * blockDim.y) {
        for(int row = blockIdx.x * blockDim.x + threadIdx.x - (w_f + 1) / 2; row < h_i; row += gridDim.x * blockDim.x) {
            if (threadIdx.x * blockDim.y + threadIdx.y < w_f * w_f) {
                shared_filter[threadIdx.x * blockDim.y + threadIdx.y] = filt_in[threadIdx.x * blockDim.y + threadIdx.y];
            }

            if (row >= 0 && row < h_i && col >= 0 && col < w_i) {
                tile[row * w_i + col] = img_in[row * w_i + col];
            } else {
                tile[row * w_i + col] = 0.0;
            }
            
            __syncthreads();

            if (row >= 0 && row < h_i && col >= 0 && col < w_i) {
                int tile_col = threadIdx.y - (w_f + 1) / 2;
                int tile_row = threadIdx.x - (w_f + 1) / 2;

                if (tile_row >= 0 && tile_row < TILE_DIM && tile_col >= 0 && tile_col < TILE_DIM ) {
                    T p_value = 0.0f;
                    for (int cc = 0; cc < w_f; cc++) {
                        for (int rr = 0; rr < w_f; rr++) {
                            p_value += shared_filter[rr * w_f + cc] * tile[(tile_row + rr) * single_input_dim_size + (tile_col + cc)];
                        }
                    }
                    img_out[row * w_i + col] = p_value;
                }
            }
        }
    }
}
#endif // BLUR_SHARED_SHARED_KERNEL_CUH