
#include <cublas_v2.h>

// Parent Headers
#include "Convolution.hpp"
#include "Matrix.hpp"
#include "Filter.hpp"
#include "BlurNaiveKernel.cuh"
#include "BlurSharedKernel.cuh"


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

  Matrix<T> filter = GetFilter<T>(height_filter_, 0, 1);
  filter.display();


  Timer ti;
  switch(switch_statement) {
  case 0:
    std::cout << "BlurNaiveKernel" << std::endl;

    T * d_filt; 
    CheckErrors(cudaMalloc((void**)&d_filt, filter.RowCnt() * filter.ColCnt() * sizeof(T)));
    CheckErrors(cudaMemcpy(d_filt, &filter(0, 0), filter.RowCnt() * filter.ColCnt() * sizeof(T), cudaMemcpyHostToDevice));

    ti.start();
    BlurNaiveKernel <<< num_blocks_per_grid, num_threads_per_block >>> (d_img_in_, d_img_out_, height_img_, width_img_, d_filt, height_filter_);
    ti.stop();


    CheckErrors(cudaMemcpy(filter.GetData(), d_img_in_, filter.RowCnt() * filter.ColCnt() * sizeof(T), cudaMemcpyDeviceToHost));
    filter.display();
    CheckErrors(cudaMemcpy(filter.GetData(), d_img_out_, filter.RowCnt() * filter.ColCnt() * sizeof(T), cudaMemcpyDeviceToHost));
    filter.display();

    CheckErrors(cudaMemcpy(filter.GetData(), d_filt, filter.RowCnt() * filter.ColCnt() * sizeof(T), cudaMemcpyDeviceToHost));
    filter.display();
    
    CheckErrors(cudaFree(d_filt));



    break;
  case 1:
    std::cout << "BlurSharedKernel" << std::endl;
    CopyFilter <<< num_blocks_per_grid, num_threads_per_block >>> (filter.GetData(), filter.ColCnt());
    

    ti.start();
    BlurSharedKernel <<< num_blocks_per_grid, num_threads_per_block >>> (d_img_in_, d_img_out_, height_img_, width_img_, height_filter_);
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
