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

    def backprop(self, output, expected):
        # calculate how sensitive the output activation is to the cost func
        activation_sensitivity = self.d_cost(output, expected)
        # loop through model, running backprop to get the gradients
        for i in range(len(self.layers) - 1, -1, -1):
            activation_sensitivity = self.layers[i].backprop(activation_sensitivity)

    def sgd(self, training_data, learn_rate, lmbda, epochs=1, validation_data=None):
        tr_x, tr_y = training_data
        previous_cost = None
        train_data_size = 0
        for t in training_data[0]:
          train_data_size += len(t)
        val_data_size = 0
        for v in validation_data[0]:
          val_data_size += len(v)

        for i in range(epochs):
            start = time.time()
            training_correct = 0
            training_total = 0
            training_cost = 0
            for batch_num in range(len(tr_x)):
                # feed data through model to calc weights, biases, activations, etc
                batch_size = tr_x[batch_num].shape[0]
                output = self.feedforward(tr_x[batch_num])
                self.backprop(output, tr_y[batch_num])
                self.update_parameters(learn_rate, lmbda, batch_size, train_data_size)

                training_correct += np.sum(np.argmax(output, axis=1) == np.argmax(tr_y[batch_num], axis=1))
                training_total += batch_size
                training_cost += np.sum(self.cost(output, tr_y[batch_num]))
                print(f"\rEpoch {i + 1}/{epochs}: ({batch_num + 1}/{len(tr_x)}) | Accuracy: {training_correct}/{training_total} | Cost: {training_cost / training_total}", end="")

            elapsed = (time.time() - start) * 1000
            print(f"\rEpoch {i + 1} Complete, Time: {round(elapsed)}ms ({round(elapsed / len(tr_x))}ms/batch), Test Accuracy: {training_correct} / {training_total}, Average Test Cost: {training_cost / training_total}", end="")
            if validation_data:
                correct, total_cost = self.evaluate(validation_data)
                avg_cost = total_cost / val_data_size
                print(f", Validation Accuracy: {correct} / {val_data_size}, Validation Average Cost = {np.round(avg_cost, 5)} {'({0:+})'.format(np.round(avg_cost - previous_cost, 5)) if previous_cost is not None else ''}", end="")
                previous_cost = avg_cost
            print()
            

    def update_parameters(self, learn_rate, lmbda, batch_size, total_size):
        # update weight and biases based on gradients
        for l in self.layers:
            total_bias_sens = np.sum(l.bias_sensitivity, axis=0)
            total_weight_sens = np.sum(l.weight_sensitivity, axis=0)
            l.biases -= learn_rate * (total_bias_sens / batch_size)
            l.weights -= learn_rate * (total_weight_sens / batch_size) + learn_rate * (lmbda / total_size) * l.weights

    def evaluate(self, validation_data):
        val_x, val_y = validation_data
        total_cost = 0
        total_correct = 0
        for i in range(len(validation_data[0])):
          out = self.feedforward(val_x[i])

          total_cost += np.sum(self.cost(out, val_y[i]))
          total_correct += np.sum(np.argmax(out, axis=1) == np.argmax(val_y[i], axis=1))

        return total_correct, total_cost

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

