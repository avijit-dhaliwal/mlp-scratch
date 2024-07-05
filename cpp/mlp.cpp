#include "mlp.h"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <iostream>

MLP::MLP(const std::vector<int>& layer_sizes) : layer_sizes(layer_sizes), gen(std::random_device{}()) {
    for (size_t i = 1; i < layer_sizes.size(); ++i) {
        std::normal_distribution<> d(0, std::sqrt(2.0 / layer_sizes[i-1]));
        weights.push_back(std::vector<std::vector<double>>(layer_sizes[i], std::vector<double>(layer_sizes[i-1])));
        for (auto& neuron_weights : weights.back()) {
            for (auto& weight : neuron_weights) {
                weight = d(gen);
            }
        }
        biases.push_back(std::vector<double>(layer_sizes[i], 0.0));
    }
}

double MLP::sigmoid(double z) {
    return 1.0 / (1.0 + std::exp(-z));
}

double MLP::sigmoid_derivative(double z) {
    double s = sigmoid(z);
    return s * (1.0 - s);
}

std::vector<std::vector<double>> MLP::forward_propagation(const std::vector<double>& x) {
    std::vector<std::vector<double>> activations = {x};
    for (size_t i = 0; i < weights.size(); ++i) {
        std::vector<double> z(weights[i].size());
        for (size_t j = 0; j < weights[i].size(); ++j) {
            z[j] = std::inner_product(weights[i][j].begin(), weights[i][j].end(), activations.back().begin(), 0.0) + biases[i][j];
        }
        activations.push_back(std::vector<double>(z.size()));
        std::transform(z.begin(), z.end(), activations.back().begin(), [this](double val) { return sigmoid(val); });
    }
    return activations;
}

std::pair<std::vector<std::vector<std::vector<double>>>, std::vector<std::vector<double>>>
MLP::backward_propagation(const std::vector<double>& x, const std::vector<double>& y) {
    auto activations = forward_propagation(x);
    std::vector<std::vector<std::vector<double>>> nabla_w(weights.size());
    std::vector<std::vector<double>> nabla_b(biases.size());

    std::vector<double> delta(activations.back().size());
    for (size_t i = 0; i < delta.size(); ++i) {
        delta[i] = (activations.back()[i] - y[i]) * sigmoid_derivative(activations.back()[i]);
    }
    nabla_b.back() = delta;

    for (size_t i = 0; i < weights.back().size(); ++i) {
        nabla_w.back().push_back(std::vector<double>(weights.back()[i].size()));
        for (size_t j = 0; j < weights.back()[i].size(); ++j) {
            nabla_w.back()[i][j] = delta[i] * activations[activations.size() - 2][j];
        }
    }

    for (int l = weights.size() - 2; l >= 0; --l) {
        std::vector<double> sp(activations[l+1].size());
        for (size_t i = 0; i < sp.size(); ++i) {
            sp[i] = sigmoid_derivative(activations[l+1][i]);
        }

        std::vector<double> new_delta(weights[l].size(), 0.0);
        for (size_t i = 0; i < weights[l].size(); ++i) {
            for (size_t j = 0; j < delta.size(); ++j) {
                new_delta[i] += weights[l+1][j][i] * delta[j];
            }
            new_delta[i] *= sp[i];
        }
        delta = new_delta;

        nabla_b[l] = delta;
        nabla_w[l] = std::vector<std::vector<double>>(weights[l].size(), std::vector<double>(weights[l][0].size()));
        for (size_t i = 0; i < weights[l].size(); ++i) {
            for (size_t j = 0; j < weights[l][i].size(); ++j) {
                nabla_w[l][i][j] = delta[i] * activations[l][j];
            }
        }
    }

    return {nabla_w, nabla_b};
}

void MLP::update_mini_batch(const std::vector<std::pair<std::vector<double>, std::vector<double>>>& mini_batch, double learning_rate) {
    std::vector<std::vector<std::vector<double>>> nabla_w(weights.size());
    std::vector<std::vector<double>> nabla_b(biases.size());

    for (size_t i = 0; i < weights.size(); ++i) {
        nabla_w[i] = std::vector<std::vector<double>>(weights[i].size(), std::vector<double>(weights[i][0].size(), 0.0));
        nabla_b[i] = std::vector<double>(biases[i].size(), 0.0);
    }

    for (const auto& [x, y] : mini_batch) {
        auto [delta_nabla_w, delta_nabla_b] = backward_propagation(x, y);

        for (size_t i = 0; i < nabla_b.size(); ++i) {
            for (size_t j = 0; j < nabla_b[i].size(); ++j) {
                nabla_b[i][j] += delta_nabla_b[i][j];
            }
        }

        for (size_t i = 0; i < nabla_w.size(); ++i) {
            for (size_t j = 0; j < nabla_w[i].size(); ++j) {
                for (size_t k = 0; k < nabla_w[i][j].size(); ++k) {
                    nabla_w[i][j][k] += delta_nabla_w[i][j][k];
                }
            }
        }
    }

    double eta = learning_rate / mini_batch.size();
    for (size_t i = 0; i < weights.size(); ++i) {
        for (size_t j = 0; j < weights[i].size(); ++j) {
            for (size_t k = 0; k < weights[i][j].size(); ++k) {
                weights[i][j][k] -= eta * nabla_w[i][j][k];
            }
            biases[i][j] -= eta * nabla_b[i][j];
        }
    }
}

void MLP::train(const std::vector<std::vector<double>>& training_data,
                const std::vector<std::vector<double>>& training_labels,
                int epochs, int mini_batch_size, double learning_rate,
                const std::vector<std::vector<double>>& test_data,
                const std::vector<std::vector<double>>& test_labels) {
    int n = training_data.size();
    std::vector<int> indices(n);
    std::iota(indices.begin(), indices.end(), 0);

    for (int epoch = 0; epoch < epochs; ++epoch) {
        std::shuffle(indices.begin(), indices.end(), gen);
        
        for (int k = 0; k < n; k += mini_batch_size) {
            std::vector<std::pair<std::vector<double>, std::vector<double>>> mini_batch;
            for (int i = k; i < std::min(k + mini_batch_size, n); ++i) {
                mini_batch.push_back({training_data[indices[i]], training_labels[indices[i]]});
            }
            update_mini_batch(mini_batch, learning_rate);
        }

        if (!test_data.empty()) {
            int correct = evaluate(test_data, test_labels);
            std::cout << "Epoch " << epoch << ": " << correct << " / " << test_data.size() << std::endl;
        } else {
            std::cout << "Epoch " << epoch << " complete" << std::endl;
        }
    }
}

std::vector<double> MLP::predict(const std::vector<double>& x) {
    return forward_propagation(x).back();
}

int MLP::evaluate(const std::vector<std::vector<double>>& test_data,
                  const std::vector<std::vector<double>>& test_labels) {
    int correct = 0;
    for (size_t i = 0; i < test_data.size(); ++i) {
        std::vector<double> prediction = predict(test_data[i]);
        if (std::distance(prediction.begin(), std::max_element(prediction.begin(), prediction.end())) ==
            std::distance(test_labels[i].begin(), std::max_element(test_labels[i].begin(), test_labels[i].end()))) {
            ++correct;
        }
    }
    return correct;
}