#include "data_loader.h"
#include <fstream>
#include <iostream>
#include <cstring>

std::vector<std::vector<double>> load_mnist_images(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open file: " + filename);
    }

    int magic_number = 0, num_images = 0, num_rows = 0, num_cols = 0;
    file.read((char*)&magic_number, sizeof(magic_number));
    file.read((char*)&num_images, sizeof(num_images));
    file.read((char*)&num_rows, sizeof(num_rows));
    file.read((char*)&num_cols, sizeof(num_cols));

    magic_number = __builtin_bswap32(magic_number);
    num_images = __builtin_bswap32(num_images);
    num_rows = __builtin_bswap32(num_rows);
    num_cols = __builtin_bswap32(num_cols);

    std::vector<std::vector<double>> images(num_images, std::vector<double>(num_rows * num_cols));
    for (int i = 0; i < num_images; ++i) {
        for (int j = 0; j < num_rows * num_cols; ++j) {
            unsigned char pixel = 0;
            file.read((char*)&pixel, sizeof(pixel));
            images[i][j] = static_cast<double>(pixel) / 255.0;
        }
    }

    return images;
}

std::vector<int> load_mnist_labels(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open file: " + filename);
    }

    int magic_number = 0, num_items = 0;
    file.read((char*)&magic_number, sizeof(magic_number));
    file.read((char*)&num_items, sizeof(num_items));

    magic_number = __builtin_bswap32(magic_number);
    num_items = __builtin_bswap32(num_items);

    std::vector<int> labels(num_items);
    for (int i = 0; i < num_items; ++i) {
        unsigned char label = 0;
        file.read((char*)&label, sizeof(label));
        labels[i] = static_cast<int>(label);
    }

    return labels;
}

std::vector<std::vector<double>> one_hot_encode(const std::vector<int>& labels, int num_classes) {
    std::vector<std::vector<double>> encoded(labels.size(), std::vector<double>(num_classes, 0.0));
    for (size_t i = 0; i < labels.size(); ++i) {
        encoded[i][labels[i]] = 1.0;
    }
    return encoded;
}