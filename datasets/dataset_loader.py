import os
import numpy as np
def load(dataset):
    if not os.path.exists(f"{dataset}.npz"):
        raise FileNotFoundError(f"Dataset \"{dataset}\" does not exist")
    return np.load(f"{dataset}.npz")

def split_batches(batch_size, *datasets):
    res = []
    for ds in datasets:
        num_batches = int(np.ceil(ds.shape[0] / batch_size))
        res.append(np.array_split(ds, num_batches))
    return res

    