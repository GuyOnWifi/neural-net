import sys
sys.path.append("../src")

from datasets import load_dataset

mnist = load_dataset('mnist')
mnist = mnist.shuffle()
print("dataset loaded")

from network import Network
from layers.dense import Dense
import numpy as np

net = Network()

net.add_layers(
    Dense((784, 80), activation="sigmoid"),
    Dense((80, 80), activation="sigmoid"),
    Dense((80, 10), activation="sigmoid"),
)

# convert the mnist data so it can work in the neural net
def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

def transform(example):
    example["label"] = [vectorized_result(x) for x in example["label"]]
    example["image"] = [np.array(img).reshape(784, 1) / 255 for img in example["image"]]
    return example

training_data = []
for data in mnist["train"]:
    training_data.append((np.array(data["image"]).reshape(784, 1) / 255, vectorized_result(data["label"])))

validation_data = []
for data in mnist["test"]:
    validation_data.append((np.array(data["image"]).reshape(784, 1) / 255, vectorized_result(data["label"])))

print("data formatted")

# train the model
net.sgd(training_data, 0.1, 32, epochs=10, validation_data=validation_data)

net.export_model("network.json")