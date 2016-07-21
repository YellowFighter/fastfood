from collections import namedtuple
import numpy as np
import scipy.special

# Holds the parameters for Fastfood kernel expansion.
FastfoodParamSet = namedtuple('FastfoodParamSet', ('B', 'G', 'PI', 'S'))


def FastfoodPara(n, d):
    """
    Create parameters for Fastfood kernel expansions.

    :param n: basis number used for Fastfood approximation
    :param d: dimension of input pattern
    :return: a FastfoodParamSet object containing the parameters for FastFood

    The FastfoodParamSet object has fields:
        - B: binary scaling matrix B in Eqn.(7)
        - G: Gaussian scaling matrix G in Eqn.(7)
        - PI: permutation matrix in Eqn.(7)
        - S: scaling matrix in Eqn.(7)
    """
    # Pad vectors with zeros until d = 2^l holds.
    l = np.ceil(np.log2(d))
    d = int(2 ** l)
    k = int(np.ceil(n / d))
    n = d * k

    B = np.zeros(shape=(k, d), dtype=int)
    G = np.zeros(shape=(k, d))
    PI = np.zeros(shape=(k, d), dtype=int)
    S = np.zeros(shape=(k, d))
    for ii in range(0, k):
        # Prepare matrix for Fastfood.
        B[ii, :] = np.random.choice([-1, 1], size=d)
        G[ii, :] = np.random.normal(size=d)
        T = np.random.permutation(d)
        PI[ii, :] = T

        # Chi distribution
        # Sampling via CDF.
        p1 = np.random.uniform(size=(d, 1))
        p2 = d / 2.0
        T = scipy.special.gammaincinv(p2, p1)
        T = np.sqrt(2*T).reshape(d)
        S[ii, :] = T * np.linalg.norm(G[ii, :].reshape((d, 1)), 'fro') ** -1.0

    S1 = np.zeros(shape=n)
    for ii in range(0, k):
        idx1 = ii*d
        idx2 = (ii+1)*d
        S1[idx1:idx2] = S[ii, :]

    params = FastfoodParamSet(B, G, PI, S1)
    return params
