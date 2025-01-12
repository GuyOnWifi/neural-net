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

        if activation == "relu":
            self.activation = self.relu
            self.d_activation = self.d_relu
        elif activation == "sigmoid":
            self.activation = self.sigmoid
            self.d_activation = self.d_sigmoid
        elif activation == "softmax":
            self.activation = self.softmax
            self.d_activation = self.d_softmax

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
        if self.activation == self.softmax:
            # tranposes along the second and third axis
            self.bias_sensitivity = np.matmul(self.d_activation(self.weighted_sums), activation_sensitivity)
        else:
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

    def softmax(self, x):
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / e_x.sum(axis=1, keepdims=True)
    
    def d_softmax(self, x):
        # jacobian matrix magic
        sm = self.softmax(x).squeeze()  # shape: (n, 10)

        # Expand sm to create a (n, 10, 1) and (n, 1, 10) for broadcasting
        sm_col = sm[:, :, np.newaxis]  # (n, 10, 1)
        sm_row = sm[:, np.newaxis, :]  # (n, 1, 10)

        # Compute Jacobian matrix for each example using broadcasting
        jacobian = sm_row * np.identity(10) - sm_col * sm_row  # (n, 10, 10)

        return jacobian
