import numpy as np

class MLP:
    def __init__(self, layer_sizes):
        self.num_layers = len(layer_sizes)
        self.layer_sizes = layer_sizes
        self.weights = [np.random.randn(y, x) * np.sqrt(2.0/x)
                        for x, y in zip(layer_sizes[:-1], layer_sizes[1:])]
        self.biases = [np.zeros((y, 1)) for y in layer_sizes[1:]]

    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def sigmoid_derivative(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))

    def forward_propagation(self, x):
        activations = [x]
        zs = []
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, activations[-1]) + b
            zs.append(z)
            activations.append(self.sigmoid(z))
        return activations, zs

    def backward_propagation(self, x, y, activations, zs):
        delta = (activations[-1] - y) * self.sigmoid_derivative(zs[-1])
        nabla_b = [delta]
        nabla_w = [np.dot(delta, activations[-2].T)]

        for l in range(2, self.num_layers):
            delta = np.dot(self.weights[-l+1].T, delta) * self.sigmoid_derivative(zs[-l])
            nabla_b.insert(0, delta)
            nabla_w.insert(0, np.dot(delta, activations[-l-1].T))

        return nabla_w, nabla_b

    def update_mini_batch(self, mini_batch, learning_rate):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in mini_batch:
            activations, zs = self.forward_propagation(x)
            delta_nabla_w, delta_nabla_b = self.backward_propagation(x, y, activations, zs)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        self.weights = [w - (learning_rate / len(mini_batch)) * nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (learning_rate / len(mini_batch)) * nb
                       for b, nb in zip(self.biases, nabla_b)]

    def train(self, training_data, epochs, mini_batch_size, learning_rate, test_data=None):
        n = len(training_data)
        for epoch in range(epochs):
            np.random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, learning_rate)
            
            if test_data:
                print(f"Epoch {epoch}: {self.evaluate(test_data)} / {len(test_data)}")
            else:
                print(f"Epoch {epoch} complete")

    def predict(self, x):
        for w, b in zip(self.weights, self.biases):
            x = self.sigmoid(np.dot(w, x) + b)
        return x

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.predict(x)), np.argmax(y)) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)