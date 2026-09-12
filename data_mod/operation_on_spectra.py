import numpy as np


class SpectraOperation:
    def __init__(self):
        pass

    def normalise(self, X, verbose=False):
        """Multiply the signal so that it takes values between 0 and 1"""
        max_intensity = np.max(X)
        if verbose:
            print(f"max intensity is {max_intensity}")
        return X / max_intensity
