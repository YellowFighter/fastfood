import numpy as np
import numpy.random as rand


def load_linear_dataset(n, d, noise_stdev, xmax):
    signs = rand.choice([-1, 1], size=(d,))
    B = np.dot(signs, rand.uniform(size=(d,)))
    errors = noise_stdev * rand.normal(size=(d,))
    X = xmax * rand.uniform(size=(d, n))
    y = np.dot(X.T, B) + errors
    return X, y, (B, errors)
