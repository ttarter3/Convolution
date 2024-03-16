
#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <vector>
#include <iostream>

template <typename T>
class Matrix {
private:
    std::vector<T> data;
    int rows;
    int cols;

public:
    // Constructor
    Matrix(int row, int col) : rows(row), cols(col) {
        data.resize(row * col, 0.0);
    }

    Matrix() {};

    // Getters
    int RowCnt() const { return rows; }
    int ColCnt() const { return cols; }

    // Overloaded () operator for accessing elements
    T& operator()(int row, int col) {
        return data[row * ColCnt() + col];
    }

    T* GetData() {
        return data.data();
    }

    std::vector<T> GetVector() {
        return data;
    }

    // Display matrix
    void display() const {
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                std::cout << data[i * ColCnt() + j] << " ";
            }
            std::cout << std::endl;
        }
    }
};

#endif // MATRIX_HPP