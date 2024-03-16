// BLUR_NAIVE_CUH
#ifndef BLUR_NAIVE_CUH
#define BLUR_NAIVE_CUH

// Cuda Headers
#include <math_constants.h>
#include <cuda_runtime.h>

template <typename T>
__global__ void BlurNaiveKernel(T * img_in, T * img_out, int h_i, int w_i, T * filter, int w_f) {
    // size_t col = blockIdx.y * blockDim.y + threadIdx.y;
    // size_t row = blockIdx.x * blockDim.x + threadIdx.x;

    for(size_t col = blockIdx.y * blockDim.y + threadIdx.y; col < w_i; col += gridDim.y * blockDim.y) {
        for(size_t row = blockIdx.x * blockDim.x + threadIdx.x; row < h_i; row += gridDim.x * blockDim.x) {
            if (col == 0 && row == 0) { printf("%d %d %d\n", w_i, h_i, w_f); }

            T p_value = 0.0f;
            for (int cc = 0; cc < w_f; cc++) {
                int in_col = col - (w_f + 1) / 2 + cc;
                for (int rr = 0; rr < w_f; rr++) {
                    int in_row = row - (w_f + 1) / 2 + rr;
                    if (in_row >= 0 && in_row < h_i && in_col >= 0 && in_col < w_i) {
                        p_value += filter[rr * w_f + cc] * img_in[in_row * w_i + in_col];
                    }
                    
                }
            }
            img_out[row * w_i + col] = img_in[row * w_i + col]; // p_value;
        }
    }
}
#endif // BLUR_NAIVE_CUH