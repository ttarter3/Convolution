
#include <cublas_v2.h>

// Parent Headers
#include "Convolution.hpp"
#include "Matrix.hpp"
#include "Filter.hpp"
#include "BlurNaiveKernel.cuh"
#include "BlurSharedKernel.cuh"
#include "BlurConstantKernel.cuh"
#include "BlurSharedSharedKernel.cuh"
#include "BlurSharedConstKernel.cuh"
#include "BlurSharedEqualConstKernel.cuh"


// Project Headers
#include "Timer.hpp"
#include "CheckError.hpp"

// Standard Headers
#include <vector>
#include <iostream>
#include <chrono>


template <typename T>
Blur<T>::Blur(int height_img, int width_img, int height_filter, int width_filter) 
  : height_img_(height_img), width_img_(width_img)
  , height_filter_(height_filter), width_filter_(width_filter) {
  // Device vectors
  CheckErrors(cudaMalloc((void**)&d_img_in_, height_img_ * width_img_ * sizeof(T)));
  CheckErrors(cudaMalloc((void**)&d_img_out_, height_img_ * width_img_ * sizeof(T)));

  filter_ = GetFilter<T>(height_filter_, 0, 3);

};

template <typename T>
Blur<T>::~Blur() {
  // Clean Up GPU
  CheckErrors(cudaFree(d_img_in_));
  CheckErrors(cudaFree(d_img_out_));
};

template <typename T>
void Blur<T>::Load(T * h_img_in) {
  // Copy host vectors to device
  CheckErrors(cudaMemcpy(d_img_in_, h_img_in, height_img_ * width_img_ * sizeof(T), cudaMemcpyHostToDevice));
  CheckErrors(cudaMemset(d_img_out_, 0, height_img_ * width_img_ * sizeof(T)));
};

template <typename T>
unsigned int Blur<T>::Execute(int tile_size, int switch_statement) {
  dim3 num_threads_per_block, num_blocks_per_grid;

  num_threads_per_block = dim3(tile_size, tile_size, 1);
  num_blocks_per_grid = dim3(
  (height_img_ + num_threads_per_block.x - 1) / num_threads_per_block.x
  , (width_img_ + num_threads_per_block.y - 1) / num_threads_per_block.y
  , 1); 
  
  std::cout << "Threads: " << num_threads_per_block.x << "," << num_threads_per_block.y << "," << num_threads_per_block.z << std::endl;
  std::cout << "Blocks:  " << num_blocks_per_grid.x << "," << num_blocks_per_grid.y << "," << num_blocks_per_grid.z << std::endl;

  
  filter_.display();


  Timer ti; T * d_filt; 
  switch(switch_statement) {
  case 0:
    std::cout << "BlurNaiveKernel" << std::endl;

    
    CheckErrors(cudaMalloc((void**)&d_filt, filter_.RowCnt() * filter_.ColCnt() * sizeof(T)));
    CheckErrors(cudaMemcpy(d_filt, &filter_(0, 0), filter_.RowCnt() * filter_.ColCnt() * sizeof(T), cudaMemcpyHostToDevice));

    ti.start();
    BlurNaiveKernel <<< num_blocks_per_grid, num_threads_per_block >>> (d_img_in_, d_img_out_, height_img_, width_img_, d_filt, height_filter_);
    ti.stop();

    // CheckErrors(cudaMemcpy(filter_.GetData(), d_img_in_, filter_.RowCnt() * filter_.ColCnt() * sizeof(T), cudaMemcpyDeviceToHost));
    // filter_.display();
    // CheckErrors(cudaMemcpy(filter_.GetData(), d_img_out_, filter_.RowCnt() * filter_.ColCnt() * sizeof(T), cudaMemcpyDeviceToHost));
    // filter_.display();

    CheckErrors(cudaMemcpy(filter_.GetData(), d_filt, filter_.RowCnt() * filter_.ColCnt() * sizeof(T), cudaMemcpyDeviceToHost));
    filter_.display();
    
    CheckErrors(cudaFree(d_filt));
    break;
  case 1:
    std::cout << "BlurSharedKernel" << std::endl;  

    CheckErrors(cudaMalloc((void**)&d_filt, filter_.RowCnt() * filter_.ColCnt() * sizeof(T)));
    CheckErrors(cudaMemcpy(d_filt, filter_.GetData(), filter_.RowCnt() * filter_.ColCnt() * sizeof(T), cudaMemcpyHostToDevice));

    ti.start();
    BlurSharedKernel <<< num_blocks_per_grid, num_threads_per_block >>> (d_img_in_, d_img_out_, height_img_, width_img_, d_filt, height_filter_);
    ti.stop();

    CheckErrors(cudaMemcpy(filter_.GetData(), d_filt, filter_.RowCnt() * filter_.ColCnt() * sizeof(T), cudaMemcpyDeviceToHost));
    filter_.display();
    
    CheckErrors(cudaFree(d_filt));

    break;

  case 2:
    std::cout << "BlurConstantKernel" << std::endl;
    
    CheckErrors(cudaMemcpyToSymbol(constant_filter, filter_.GetData(), filter_.RowCnt() * filter_.ColCnt() * sizeof(float)));
    ti.start();
    // d_filt is not necessary but I like the consistency
    BlurConstantKernel <<< num_blocks_per_grid, num_threads_per_block >>> (d_img_in_, d_img_out_, height_img_, width_img_, d_filt, height_filter_); 
    ti.stop();

    break;
  
  case 3:
    std::cout << "BlurSharedSharedKernel" << std::endl;
    
    num_threads_per_block = dim3(tile_size + (height_filter_ + 1) / 2 * 2, tile_size + (height_filter_ + 1) / 2 * 2, 1);
    num_blocks_per_grid = dim3(
    (height_img_ + tile_size - 1) / tile_size
    , (width_img_ + tile_size - 1) / tile_size
    , 1); 

    ti.start();
    BlurSharedSharedKernel <<< num_blocks_per_grid, num_threads_per_block >>> (d_img_in_, d_img_out_, height_img_, width_img_, d_filt, height_filter_);
    ti.stop();

    break;

  case 4:
    std::cout << "BlurSharedConstKernel" << std::endl;
    
    CheckErrors(cudaMemcpyToSymbol(constant_filter, filter_.GetData(), filter_.RowCnt() * filter_.ColCnt() * sizeof(float)));

    num_threads_per_block = dim3(tile_size + (height_filter_ + 1) / 2 * 2, tile_size + (height_filter_ + 1) / 2 * 2, 1);
    num_blocks_per_grid = dim3(
    (height_img_ + tile_size - 1) / tile_size
    , (width_img_ + tile_size - 1) / tile_size
    , 1); 

    ti.start();
    BlurSharedConstKernel <<< num_blocks_per_grid, num_threads_per_block >>> (d_img_in_, d_img_out_, height_img_, width_img_, d_filt, height_filter_);
    ti.stop();

    break;

  case 5:
    std::cout << "BlurSharedEqualConstKernel" << std::endl;
    
    CheckErrors(cudaMemcpyToSymbol(constant_filter, filter_.GetData(), filter_.RowCnt() * filter_.ColCnt() * sizeof(float)));

    num_threads_per_block = dim3(tile_size + (height_filter_ + 1) / 2 * 2, tile_size + (height_filter_ + 1) / 2 * 2, 1);
    num_blocks_per_grid = dim3(
    (height_img_ + tile_size - 1) / tile_size
    , (width_img_ + tile_size - 1) / tile_size
    , 1); 

    ti.start();
    BlurSharedEqualConstKernel <<< num_blocks_per_grid, num_threads_per_block >>> (d_img_in_, d_img_out_, height_img_, width_img_, d_filt, height_filter_);
    ti.stop();

    break;

  default:
    std::cout << "ERROROROROROR" << std::endl;
    exit(1);
  }

  

  std::cout << "RUN TIME(ms): " << ti.elapsedTime_ms() << std::endl;
  return  (unsigned int) (ti.elapsedTime_ms() * 1000.0);
};

template <typename T>
void Blur<T>::Unload(T * h_img_out) {
  // Copy result back to host
  CheckErrors(cudaMemcpy(h_img_out, d_img_out_, height_img_ * width_img_ * sizeof(T), cudaMemcpyDeviceToHost));      
};

template class Blur<float>;
template class Blur<double>;
