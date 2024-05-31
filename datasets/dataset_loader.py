import os
import numpy as np
def load(dataset):
    dirname = os.path.dirname(os.path.realpath(__file__))
    dataset_file = os.path.join(dirname, f"{dataset}.npz")
    if not os.path.exists(dataset_file):
        raise FileNotFoundError(f"Dataset \"{dataset}\" does not exist")
    return np.load(dataset_file)

def split_batches(batch_size, *datasets):
    res = []
    for ds in datasets:
        num_batches = int(np.ceil(ds.shape[0] / batch_size))
        res.append(np.array_split(ds, num_batches))
    return res

    