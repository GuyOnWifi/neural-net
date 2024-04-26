import numpy as np
import json
import time

class Network:
    def __init__(self):
        self.layers = []
        self.size = []
    
    def add_layers(self, *layers):
        for l in layers:
            self.layers.append(l)
            if len(self.size) != 0:
                self.size.pop()
            self.size.append(l.shape[0])
            self.size.append(l.shape[1])
    
    def feedforward(self, input):
        for l in self.layers:
            input = l.feedforward(input)
        return input

    def backprop(self, input, expected):
        # feed data through model to calc weights, biases, activations, etc
        output = self.feedforward(input)
        # calculate how sensitive the output activation is to the cost func
        activation_sensitivity = self.d_cost(output, expected)
        # loop through model, running backprop to get the gradients
        for i in range(len(self.layers) - 1, -1, -1):
            activation_sensitivity = self.layers[i].backprop(activation_sensitivity)

    def sgd(self, training_data, learn_rate, batch_size, epochs=1, validation_data=None):
        mini_batches = [training_data[k:k + batch_size] for k in range(0, len(training_data), batch_size)]
        
        previous_cost = None
        for i in range(epochs):
            start = time.time()
            batch_num = 0
            for batch in mini_batches:
                self.fit(batch, learn_rate)
                batch_num += 1
                print(f"\rEpoch {i + 1}/{epochs}: ({batch_num}/{len(mini_batches)})", end="")

            if validation_data:
                elapsed = (time.time() - start) * 1000
                correct, avg_cost = self.evaluate(validation_data)
                print(f"\rEpoch {i + 1}/{epochs}: {correct} / {len(validation_data)}, Time: {round(elapsed)}ms ({round(elapsed / len(mini_batches))}ms/batch), Average Cost = {round(avg_cost, 5)} {'({0:+})'.format(round(avg_cost - previous_cost, 5)) if previous_cost is not None else ''}")
                previous_cost = avg_cost
            
    def fit(self, batch, learn_rate):
        # reset variables
        for l in self.layers:
            l.total_bias_sens *= 0
            l.total_weight_sens *= 0

        for inp, exp in batch:
            self.backprop(inp, exp)

            # sum up total sensitivity
            for l in self.layers:
                l.total_bias_sens += l.bias_sensitivity
                l.total_weight_sens += l.weight_sensitivity
                
        # adjust the weights and biases
        for l in self.layers:
            l.biases -= learn_rate * (l.total_bias_sens / len(batch))
            l.weights -= learn_rate * (l.total_weight_sens / len(batch))

    def evaluate(self, validation_data):
        total_cost = 0
        total_correct = 0
        for inp, exp in validation_data:
            out = self.feedforward(inp)
            total_cost += np.sum(self.cost(out, exp))
            if np.argmax(out) == np.argmax(exp):
                total_correct += 1
        return total_correct, total_cost / len(validation_data)

    def cost(self, output, expected):
        return np.square(output - expected)

    def d_cost(self, output, expected):
        return 2 * (output - expected)

    def load_model(self, filename):
        f = open(filename, "r")
        data = json.load(f)
        f.close()
        if (self.size != data["sizes"]):
            raise TypeError(f"Data does not match network size of {self.size}")

        for i in range(len(data["weights"])):
            w = np.array(data["weights"][i])
            self.layers[i].weights = w

        for i in range(len(data["biases"])):
            w = np.array(data["biases"][i])
            self.layers[i].biases = w
    
    def export_model(self, filename):
        data = {
            "sizes": self.size,
            "weights": [l.weights.tolist() for l in self.layers],
            "biases": [l.biases.tolist() for l in self.layers]
        }

        f = open(filename, "w")
        json.dump(data, f)
        f.close()

