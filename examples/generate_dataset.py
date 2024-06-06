from datasets import load_dataset

mnist = load_dataset('fashion_mnist')

import numpy as np

def convert_dataset(data):
    arr = np.zeros((len(data), 784, 1))
    i = 0
    for d in data:
        arr[i] = np.array(d).reshape((784, 1))
        i += 1
    return arr

def convert_answers(data):
    arr = np.zeros((len(data), 10, 1))
    i = 0
    for d in data:
        e = np.zeros(10)
        e[d] = 1.0
        arr[i] = e.reshape(10, 1)
        i += 1
    return arr

train_data = convert_dataset(mnist["train"]["image"])
train_ans = convert_answers(mnist["train"]["label"])
test_data = convert_dataset(mnist["test"]["image"])
test_ans = convert_answers(mnist["test"]["label"])

print(train_data.shape, train_ans.shape, test_data.shape, test_ans.shape)

np.savez_compressed(
    "fashion_mnist", 
    train_data = train_data,
    train_ans = train_ans,
    test_data = test_data,
    test_ans = test_ans
)