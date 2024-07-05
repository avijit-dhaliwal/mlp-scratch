#include "mlp.h"
#include "data_loader.h"
#include <iostream>
#include <chrono>
#include <random>
#include <algorithm>
#include <vector>
#include <string>
#include <nlohmann/json.hpp>

// Simple JSON writer functions
std::string to_json(const std::string& key, const std::string& value) {
    return "\"" + key + "\": \"" + value + "\"";
}
std::string to_json(const std::string& key, double value) {
    return "\"" + key + "\": " + std::to_string(value);
}
std::string to_json(const std::string& key, const std::vector<double>& values) {
    std::string result = "\"" + key + "\": [";
    for (size_t i = 0; i < values.size(); ++i) {
        result += std::to_string(values[i]);
        if (i < values.size() - 1) result += ", ";
    }
    result += "]";
    return result;
}

int main() {
    // Set hyperparameters
    std::vector<int> layer_sizes = {784, 128, 64, 10};
    double learning_rate = 0.1;
    int epochs = 30;
    int mini_batch_size = 32;

    // Set fixed seed for reproducibility
    std::mt19937 gen(42);

    std::cout << "Loading MNIST data..." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();

    // Load and preprocess data
    download_mnist();  // Ensure the dataset is downloaded
    auto train_images = load_mnist_images("train-images-idx3-ubyte");
    auto train_labels = load_mnist_labels("train-labels-idx1-ubyte");
    auto test_images = load_mnist_images("t10k-images-idx3-ubyte");
    auto test_labels = load_mnist_labels("t10k-labels-idx1-ubyte");

    auto train_labels_encoded = one_hot_encode(train_labels);
    auto test_labels_encoded = one_hot_encode(test_labels);

    // Shuffle training data
    std::vector<size_t> indices(train_images.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), gen);

    std::cout << "Initializing MLP..." << std::endl;
    MLP mlp(layer_sizes);

    std::cout << "Training MLP..." << std::endl;
    std::vector<double> epoch_times;
    std::vector<double> accuracies;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        auto epoch_start = std::chrono::high_resolution_clock::now();

        // Train on mini-batches
        for (size_t i = 0; i < train_images.size(); i += mini_batch_size) {
            size_t batch_size = std::min(mini_batch_size, train_images.size() - i);
            std::vector<std::vector<double>> batch_images(batch_size);
            std::vector<std::vector<double>> batch_labels(batch_size);

            for (size_t j = 0; j < batch_size; ++j) {
                batch_images[j] = train_images[indices[i + j]];
                batch_labels[j] = train_labels_encoded[indices[i + j]];
            }

            mlp.train(batch_images, batch_labels, learning_rate);
        }

        auto epoch_end = std::chrono::high_resolution_clock::now();
        double epoch_time = std::chrono::duration<double>(epoch_end - epoch_start).count();
        epoch_times.push_back(epoch_time);

        // Evaluate on test set
        int correct = mlp.evaluate(test_images, test_labels_encoded);
        double accuracy = static_cast<double>(correct) / test_images.size();
        accuracies.push_back(accuracy);

        std::cout << "Epoch " << epoch + 1 << "/" << epochs 
                  << ", Time: " << epoch_time << "s"
                  << ", Accuracy: " << accuracy << std::endl;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    double training_time = std::chrono::duration<double>(end_time - start_time).count();

    // Output results as JSON
    nlohmann::json output;
    output["training_time"] = training_time;
    output["accuracy"] = accuracies.back();
    output["epoch_times"] = epoch_times;
    output["accuracies"] = accuracies;
    std::cout << output.dump() << std::endl;

    return 0;
}