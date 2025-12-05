import numpy as np

def harmonic_mean(array):
    """
    Helper fn to calculate the harmonic mean of a 1D numpy array.
    """
    n = len(array)
    return n / np.sum(1.0 / array)