import sys
sys.path.append("../datasets")
from dataset_loader import load, split_batches
        
data = load("mnist")
train_data, train_ans, test_data, test_ans = data.values()
train_data /= 255
test_data /= 255
train_data, train_ans = split_batches(256, train_data, train_ans)
test_data, test_ans = split_batches(100, test_data, test_ans)
print("dataset loaded")

sys.path.append("../src")
from network import Network
from layers.dense import Dense

net = Network()

net.add_layers(
    Dense((784, 80), activation="sigmoid"),
    Dense((80, 80), activation="sigmoid"),
    Dense((80, 10), activation="sigmoid"),
)

#net.load_model("network.json")

net.sgd((train_data, train_ans), 3, 0.01, epochs=10, validation_data=(test_data, test_ans))

net.export_model("network.json")