#pragma once
#include <vector>
#include <random>

class MLP {
public:
    MLP(const std::vector<int>& layer_sizes);
    void train(const std::vector<std::vector<double>>& training_data,
               const std::vector<std::vector<double>>& training_labels,
               int epochs, int mini_batch_size, double learning_rate,
               const std::vector<std::vector<double>>& test_data,
               const std::vector<std::vector<double>>& test_labels);
    std::vector<double> predict(const std::vector<double>& x);
    int evaluate(const std::vector<std::vector<double>>& test_data,
                 const std::vector<std::vector<double>>& test_labels);

private:
    std::vector<int> layer_sizes;
    std::vector<std::vector<std::vector<double>>> weights;
    std::vector<std::vector<double>> biases;
    std::mt19937 gen;

    double sigmoid(double z);
    double sigmoid_derivative(double z);
    std::vector<std::vector<double>> forward_propagation(const std::vector<double>& x);
    std::pair<std::vector<std::vector<std::vector<double>>>, std::vector<std::vector<double>>>
    backward_propagation(const std::vector<double>& x, const std::vector<double>& y);
    void update_mini_batch(const std::vector<std::pair<std::vector<double>, std::vector<double>>>& mini_batch,
                           double learning_rate);
};