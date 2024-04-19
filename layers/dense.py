import numpy as np

class Dense():
    def __init__(self, shape, activation="relu"):
        self.shape = shape
        self.biases = np.random.randn(shape[1], 1)
        self.weights = np.random.randn(shape[1], shape[0])
        self.weighted_sums = []
        self.inputs = []
        self.bias_sensitivity = []
        self.weight_sensitivity = []

        self.total_weight_sens = np.zeros((shape[1], shape[0]))
        self.total_bias_sens = np.zeros((shape[1], 1))


    def feedforward(self, inputs):
        self.inputs = inputs
        self.weighted_sums = np.dot(self.weights, inputs) + self.biases
        return self.sigmoid(self.weighted_sums)

    def backprop(self, activation_sensitivity):
        # partial derivatives magic
        # calc how sensitive the bias is (calculate the sigmoid deriv and multiply by how sensitive the output activations are)
        self.bias_sensitivity = activation_sensitivity * self.d_sigmoid(self.weighted_sums)
        # calc how sensitive weights are, input is transposed for matrix multiplication purposes
        self.weight_sensitivity = np.dot(self.bias_sensitivity, self.inputs.transpose())
        # return the sensitivity of previous activation, will be used in next layer
        return np.dot(self.weights.transpose(), self.bias_sensitivity)

    
    def relu(self, inp):
        return inp * (inp > 0)
    
    def d_relu(self, inp):
        return 1. * (inp > 0)

    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))
    
    def d_sigmoid(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))

    