import numpy as np


def pad_to_power_of_2(matrix):
    # TODO find a way to modify matrix rather than making a copy
    orig = matrix.shape[0]
    next_log2 = 2**(int(np.floor(np.log2(orig))) + 1)
    new_matrix = np.zeros((next_log2, matrix.shape[1]), dtype=matrix.dtype)
    new_matrix[:orig, :] = matrix
    return new_matrix
