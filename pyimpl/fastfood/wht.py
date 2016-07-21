import scipy.linalg
import numpy as np

__all__ = ['WHT', 'FWHT']


def Hadamard2Walsh(n):
    hadamardMatrix = scipy.linalg.hadamard(n)
    hadIdx = np.arange(n)
    M = np.log2(n) + 1

    binHadIdx = None
    for i in hadIdx:
        s = format(i, '#032b')
        s = s[::-1]
        s = s[:-2]
        s = list(s)
        x = [int(x) for x in s]
        x = np.array(x)
        if i == 0:
            binHadIdx = x
        else:
            binHadIdx = np.vstack((binHadIdx, x))
        binSeqIdx = np.zeros(shape=(n, M)).T

        for k in reversed(range(1, int(M))):
            tmp = np.bitwise_or(binHadIdx.T[k], binHadIdx.T[k - 1])
            binSeqIdx[k] = tmp

        tmp = np.power(2, np.arange(M)[::-1])
        tmp = tmp.T
        seqIdx = np.dot(binSeqIdx.T, tmp)

        j = 1
        for i in seqIdx:
            if j == 1:
                walshMatrix = hadamardMatrix[i]
            else:
                walshMatrix = np.vstack((walshMatrix,
                                         hadamardMatrix[i]))
            j += 1

        return hadamardMatrix, walshMatrix


def WHT(x):
    """
    Computes (slow) Discrete Walsh-Hadamard Transform for
    any 1D real-valued signal.
    :param x: 1D real-valued signal
    :return: 1D real-valued transformed signal, of a size a power of 2
    """
    x = np.array(x)
    if len(x.shape) < 2:  # Make sure x is 1D.
        if len(x) > 3:  # Accept x of min length of 4 elems (M=2)
            # Check length of signal, adjust to 2**m
            n = len(x)
            M = int(np.trunc(np.log2(n)))
            x = x[:2 ** M]
            h2 = np.array([[1, 1], [1, -1]])
            for i in range(M - 1):
                if i == 0:
                    H = np.kron(h2, h2)
                else:
                    H = np.kron(H, h2)
            tmp = np.dot(H, x) / 2.0 ** M
            return tmp, x, M
        else:
            raise ValueError('HWT(x): Array too short, min length 4.')
    else:
        raise ValueError('HWT(x): 1D array required.')


def bit_reverse_traverse(a):
    # (c) 2014 Ryan Compton
    # ryancompton.net/2014/06/05/bit-reversal-permutation-in-python/
    n = a.shape[0]
    if n & (n - 1):  # assert that n is a power of 2
        raise ValueError('Size of first dimension of a must be a multiple of 2.')
    if n == 1:
        yield a[0]
    else:
        even_index = np.arange(n / 2, dtype=int) * 2
        odd_index = np.arange(n / 2, dtype=int) * 2 + 1
        for even in bit_reverse_traverse(a[even_index]):
            yield even
        for odd in bit_reverse_traverse(a[odd_index]):
            yield odd


def get_bit_reversed_list(l):
    # (c) 2014 Ryan Compton
    # ryancompton.net/2014/06/05/bit-reversal-permutation-in-python/
    n = len(l)
    indexs = np.arange(n)
    b = []
    for i in bit_reverse_traverse(indexs):
        b.append(l[i])
    return b


def FWHT(X):
    # Fast Walsh-Hadamard Transform for 1D signals
    # of length n=2^M only (non error-proof for now)
    x = get_bit_reversed_list(X)
    x = np.array(x)
    N = len(X)

    for i in range(0, N, 2):
        x[i] = x[i] + x[i + 1]
        x[i + 1] = x[i] - 2 * x[i + 1]

    L = 1
    y = np.zeros_like(x)
    for n in range(2, int(np.log2(N)) + 1):
        M = 2 ** L
        J = 0
        K = 0
        while K < N:
            for j in range(J, J + M, 2):
                y[K] = x[j] + x[j + M]
                y[K + 1] = x[j] - x[j + M]
                y[K + 2] = x[j + 1] + x[j + 1 + M]
                y[K + 3] = x[j + 1] - x[j + 1 + M]
                K += 4
            J += 2 * M
        x = y.copy()
        L += 1

    return x / float(N)
