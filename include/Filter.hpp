
#ifndef FILTER_HPP
#define FILTER_HPP

// Project Headers
#include <Matrix.hpp>
#include <Constants.hpp>

// Standard Headers
#include <vector>
#include <cmath>

template <typename T>
Matrix<T> GetFilter(int N, T mean_x, T std_x) {
    Matrix<T> filt(N, N);
    
    Constants const_;

    for (int ii = 0; ii < N; ii++) {
        for (int jj = 0 ; jj < N; jj++) {
            T x = 2 * std_x / (N - 1) * ii - std_x;
            T y = 2 * std_x / (N - 1) * jj - std_x;
            filt(ii, jj) = 1 / ( 2 * const_.PI() * std::pow(std_x, 2) );
            filt(ii, jj) *= std::exp(-1 * std::pow(x, 2) + std::pow(y, 2) / (2 * std::pow(std_x, 2)));
        }
    }

    return filt;
}

#endif // FILTER_HPP