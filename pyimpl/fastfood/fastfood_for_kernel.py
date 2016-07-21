import numpy as np
from .wht import FWHT
__all__ = ['FastfoodForKernel']


def FastfoodForKernel(X, para, sgm):
    """
    Perform Fastfood kernel expansions.

    :param X: input patterns, each column is an input pattern
    :param para: parameters for Fastfood
    :param sgm: bandwidth for Gaussian kernel
    :param use_spiral: whether to use Spiral package to perform Walsh-Hadamard transform
    :return: (PHI, THT) where phi is the feature map in Eqn.(8),
            and tht are the angles used for feature mapping, V*x*sgm, where V
            is the same as that in Eqn.(8).

    """
    # Pad the vectors with zeros until d = 2^l holds.
    d0, m = X.shape
    l = np.ceil(np.log2(d0))
    d = int(2 ** l)
    if d != d0:
        XX = np.zeros(shape=(d, m))
        XX[0:d0, :] = X
        X = XX

    k = para.B.shape[0]
    n = d * k
    THT = np.zeros(shape=(n, m))

    for ii in range(0, k):
        B = para.B[ii, :]
        G = para.G[ii, :]
        PI = para.PI[ii, :]
        X = np.multiply(X.T, B).T
        T = FWHT(X)
        T = T[PI, :]
        T = np.multiply(X.T, G*d).T
        idx1 = ii*d
        idx2 = (ii+1)*d
        THT[idx1:idx2, :] = FWHT(T)

    S = para.S
    THT = np.multiply(THT.T, S*np.sqrt(d)).T

    T = THT / sgm
    PHI = np.vstack((np.cos(T), np.sin(T))) * (1.0/np.sqrt(n))

    return PHI, THT
