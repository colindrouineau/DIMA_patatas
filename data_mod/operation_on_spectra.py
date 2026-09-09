import numpy as np


class SpectraOperation:
    def __init__(self):
        pass

    def normalise(self, X):
        if len(X.shape) == 1:
            return (X - np.mean(X, axis=0, keepdims=True)) / np.std(
                X, axis=0, keepdims=True
            )
        else:
            return (X - np.mean(X, axis=1, keepdims=True)) / np.std(
                X, axis=1, keepdims=True
            )
