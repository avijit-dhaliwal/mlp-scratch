#pragma once
#include <vector>
#include <string>

void download_mnist();
std::vector<std::vector<double>> load_mnist_images(const std::string& filename);
std::vector<int> load_mnist_labels(const std::string& filename);
std::vector<std::vector<double>> one_hot_encode(const std::vector<int>& labels, int num_classes = 10);