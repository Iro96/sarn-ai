import numpy as np
from .functions import neuro_spike

class SARNLayer:
    def __init__(self, input_size, output_size, k=3, activation=neuro_spike):
        self.input_size = input_size
        self.output_size = output_size
        self.k = k
        self.activation = activation
        self.weights = np.random.randn(output_size, input_size) * 0.1
        self.biases = np.zeros(output_size)
        self.last_input = None
        self.last_output = None

    def forward(self, x):
        x = np.array(x)
        raw = self.weights @ x + self.biases
        activated = self.activation(raw)
        top_k_idx = np.argpartition(activated, -self.k)[-self.k:]
        result = np.zeros_like(activated)
        result[top_k_idx] = activated[top_k_idx]
        self.last_input = x
        self.last_output = result
        return result

    def add_neuron(self):
        new_weights = np.random.randn(1, self.input_size) * 0.1
        self.weights = np.vstack([self.weights, new_weights])
        self.biases = np.append(self.biases, 0.0)
        self.output_size += 1

    def set_weights(self, weights):
        self.weights = np.array(weights)

    def set_biases(self, biases):
        self.biases = np.array(biases)
