import numpy as np
class Dense():
    def __init__(self, shape, activation=None):
        self.shape = shape
        self.biases = np.zeros((shape[1], 1))
        self.weights = np.random.randn(shape[1], shape[0]) / np.sqrt(shape[0] / 2)
        self.weighted_sums = []
        self.inputs = []
        self.bias_sensitivity = []
        self.weight_sensitivity = []

        if (activation == 'relu'):
            self.activation = self.relu
            self.d_activation = self.d_relu
        elif (activation == 'sigmoid'):
            self.activation = self.sigmoid
            self.d_activation = self.d_sigmoid

    def activation(self, inp):
        return inp

    def d_activation(self, inp):
        return inp

    def feedforward(self, inputs):
        self.inputs = inputs
        self.weighted_sums = np.matmul(self.weights[np.newaxis, ...], inputs) + self.biases[np.newaxis, ...]
        return self.activation(self.weighted_sums)

    def backprop(self, activation_sensitivity):
        # partial derivatives magic
        # calc how sensitive the bias is (calculate the sigmoid deriv and multiply by how sensitive the output activations are)
        self.bias_sensitivity = activation_sensitivity * self.d_activation(self.weighted_sums)
        # calc how sensitive weights are, input is transposed for matrix multiplication purposes
        self.weight_sensitivity = np.matmul(self.bias_sensitivity, self.inputs.transpose([0, 2, 1]))
        # return the sensitivity of previous activation, will be used in next layer
        return np.matmul(self.weights.transpose()[np.newaxis, ...], self.bias_sensitivity)

    def relu(self, inp):
        return inp * (inp > 0)

    def d_relu(self, inp):
        return 1. * (inp > 0)

    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def d_sigmoid(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))

