#ifndef CONVOLUTION_HPP
#define CONVOLUTION_HPP

#include <Matrix.hpp>

// Standard Headers
#include <vector>



template<typename T>
class Blur {
  public:
    Blur(int height_img, int width_img, int height_filter, int width_filter);
    ~Blur();

    void Load(T * h_img_in);
    unsigned int Execute(int tile_size, int switch_statement);
    void Unload(T * h_img_out);
    
    Matrix<T> GetMatrix() {
      return filter_;
    }
  private:
    int height_img_;
    int width_img_;
    int height_filter_;
    int width_filter_;
    
    T * d_img_in_;
    T * d_img_out_;

    Matrix<T> filter_;
};


#endif // CONVOLUTION_HPP