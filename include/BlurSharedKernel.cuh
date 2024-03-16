// SHARED_BLUR_CUH
#ifndef SHARED_BLUR_CUH
#define SHARED_BLUR_CUH

// Cuda Headers
#include <math_constants.h>
#include <cuda_runtime.h>


#define CONSTANT_SIZE 768
template<typename T>
__shared__ T shared_filter[CONSTANT_SIZE];

template<typename T>
__global__ void CopyFilter(T * filt_in, int w_f) {
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
        for(int ii = 0 ; ii < w_f * w_f ; ii++) {
            shared_filter<T>[ii] = filt_in[ii];
            // printf("%d -> %f\n", ii, shared_filter[ii]);
        }
    }
    __syncthreads();
}

template<typename T>
__global__ void BlurSharedKernel(T * img_in, T * img_out, int h_i, int w_i, int w_f) {
    for(size_t col = blockIdx.y * blockDim.y + threadIdx.y; col < w_i; col += gridDim.y * blockDim.y) {
        for(size_t row = blockIdx.x * blockDim.x + threadIdx.x; row < h_i; row += gridDim.x * blockDim.x) {
            T p_value = 0.0f;
            for (int cc = 0; cc < w_f; cc++) {
                int in_col = col - (w_f + 1) / 2 + cc;
                for (int rr = 0; rr < w_f; rr++) {
                    int in_row = row - (w_f + 1) / 2 + rr;
                    if (in_row >= 0 && in_row < h_i && in_col >= 0 && in_col < w_i) {
                        p_value += shared_filter<T>[rr * w_f + cc] * img_in[in_row * w_i + in_col];
                    }
                }
            }
            img_out[row * w_i + col] = static_cast<T>(p_value);
        }
    }
}
#endif // SHARED_BLUR_CUH