import numpy.random as np
import numpy as np


def train_test_inxs(n, train_frac):
    test_inxs = rand.choice(n, int(p * (1.0 - train_frac)))
    mask = np.zeros(data.shape[0], dtype=bool)
    mask[test_inxs] = True
    test_inxs = np.arange(n)[mask]
    train_inxs = np.arange(n)[~mask]
    return train_inxs, test_inxs


def train_test_split(data, train_frac):
    train_inxs, test_inxs = train_test_inxs(data.shape[0], train_frac)
    return data[train_inxs], data[test_inxs], (train_inxs, test_inxs)
