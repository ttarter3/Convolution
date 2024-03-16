

#include <iostream>

#include "Convolution.hpp"

// 3rd Party Headers
#include <cblas.h> 


// Standard Header
#include <fstream>
#include <iterator>
#include <iterator>
#include <algorithm>
#include <string>
#include <vector>
#include <random>
#include <iomanip>
#include <chrono>
#include <numeric>


std::vector<float> ReadFile(const std::string& filename) {
    std::vector<float> floats;
    std::ifstream file(filename, std::ios::binary);

    if (!file.is_open()) {
        std::cerr << "Error: Unable to open file " << filename << std::endl;
        return floats;
    }

    // Find the length of the file
    file.seekg(0, std::ios::end);
    std::streampos fileSize = file.tellg();
    file.seekg(0, std::ios::beg);

    // Calculate the number of floats
    size_t numFloats = fileSize / sizeof(float);

    // Resize the vector to hold all floats
    floats.resize(numFloats);

    // Read the floats into the vector
    file.read(reinterpret_cast<char*>(&floats[0]), fileSize);

    // Check if reading was successful
    if (!file) {
        std::cerr << "Error: Unable to read from file " << filename << std::endl;
        floats.clear(); // Clear the vector
    }

    file.close();
    return floats;
}

void WriteFile(std::string& filename, std::vector<float>& floats) {
    std::ofstream file(filename, std::ios::binary);

    if (!file.is_open()) {
        std::cerr << "Error: Unable to open file " << filename << std::endl;
        return;
    }

    // Write the floats to the file
    file.write(reinterpret_cast<const char*>(floats.data()), floats.size() * sizeof(float));

    // Check if writing was successful
    if (!file) {
        std::cerr << "Error: Unable to write to file " << filename << std::endl;
    }

    file.close();
}

int main(int argc, char* argv[]) {
    // Your main function code goes here
    std::cout << "Start" << std::endl;
    
    unsigned int tile_size = 32;

    std::string input_file("./reference/grey_lin_wagon.bin");
    std::vector<float> in_img = ReadFile(input_file);

    for (int ii = 0 ; ii < 100; ii++) {
      std::cout << in_img[ii] << ", ";
    } std::cout << std::endl;

    std::vector<float> times(6);

    for (int ii = 0; ii < 6; ii++) {
        std::vector<float> out_img(in_img.size(), 0);

        int N = 3; 
        Blur<float> blur(1080, 1920, N, N);
        blur.Load(in_img.data());

        for (int jj = 0 ; jj < 25; jj++) {
        std::cout << in_img[jj] << ", ";
        } std::cout << std::endl;

        // blur.Execute(tile_size, 0);
        times[ii] = blur.Execute(tile_size, 3);

        blur.Unload(out_img.data());

        for (int jj = 0 ; jj < 25; jj++) {
        std::cout << out_img[jj] << ", ";
        } std::cout << std::endl;

        std::string output_file_fil("./data/Filter." + std::to_string(ii) + ".bin");
        std::vector<float> fil = blur.GetMatrix().GetVector(); 
        WriteFile(output_file_fil, fil);

        std::string output_file_img("./data/Image." + std::to_string(ii) + ".bin");
        WriteFile(output_file_img, out_img);
    }

    std::string output_file_time("./data/Time.bin");
    WriteFile(output_file_time, times);

    std::cout << "Finish" << std::endl;
    return 0;
}
