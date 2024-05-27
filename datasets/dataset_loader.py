import os
import numpy as np
def load(dataset):
    if not os.path.exists(f"{dataset}.npz"):
        raise FileNotFoundError(f"Dataset \"{dataset}\" does not exist")
    return np.load(f"{dataset}.npz")