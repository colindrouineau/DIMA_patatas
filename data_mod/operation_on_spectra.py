import numpy as np


class SpectraOperation:
    def __init__(self):
        pass

    def normalise(self, X):
        axis = len(X.shape) - 1
        return (X - np.mean(X, axis=axis, keepdims=True)) / np.std(
            X, axis=axis, keepdims=True
        )
