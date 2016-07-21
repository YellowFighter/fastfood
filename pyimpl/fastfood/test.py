import numpy as np
from unittest import main, TestCase
from fastfood import FastfoodPara, FastfoodForKernel
from fastfood.wht import WHT, FWHT


class WHTTestBase:
    input = [0.06585913, -0.18498446, 0.83443832, -0.49905254, -0.49782276,
             0.30332424, 0.57375095, 1.61769819, 1.63629352, -0.76197603,
             -1.58895883, 0.271738, 0.14624975, -1.30765071, 0.69257974,
             -0.05697602]
    output = [0.07778191, 0.15501682, -0.15287032, 0.2577165, -0.10611227,
              0.11022158, 0.36999872, 0.13932339, 0.19886948, -0.18761179,
              -0.20218703, -0.36269738, -0.116474, 0.31845701, -0.12856915,
              -0.30500432]


class WHTTest(TestCase, WHTTestBase):
    def test_wht(self):
        actual = WHT(self.input)
        map(lambda i: self.assertEquals(actual[i], self.output[i]),
            range(len(actual)))

    def test_too_short(self):
        malformed_input = [1, -1]
        try:
            WHT(malformed_input)
        except ValueError:
            pass
        else:
            raise Exception('WHT accepted vector shorter than length 4.')


class FWHTTest(TestCase, WHTTestBase):
    def test_fwht(self):
        actual = FWHT(self.input)
        map(lambda i: self.assertEquals(actual[i], self.output[i]),
            range(len(actual)))

    def test_not_multiple_2(self):
        malformed_input = [1, -1, 1]
        try:
            FWHT(malformed_input)
        except ValueError:
            pass
        else:
            raise Exception(
                'FWHT accepted vector whose length was not a multiple of 2.')


class FFForKernelTest(TestCase):
    acc = 0.98415801223151255783250235253945

    def test_k_appro(self):
        d = 64  # Dimension of input pattern
        n = d * 20  # Basis number used for approximation
        sgm = 10  # Bandwidth for Gaussian kernel
        N = 10  # Number of sample input patterns to generate

        np.random.seed(0)  # For reproducibility.

        X1 = np.random.randn(d, N)
        X2 = np.random.randn(d, int(1.5 * N))

        # Exact calculation of Gaussian kernel
        K_exact = np.zeros(shape=(X1.shape[1], X2.shape[1]))
        for i in range(X1.shape[1]):
            for j in range(X2.shape[1]):
                K_exact[i, j] = np.exp(
                    -np.linalg.norm(X1[:, i] - X2[:, j], 2) / (2 * sgm ** 2)
                )

        # Fastfood approximation of Gaussian kernel
        para = FastfoodPara(n, d)
        PHI1, THT1 = FastfoodForKernel(X1, para, sgm)
        PHI2, THT2 = FastfoodForKernel(X2, para, sgm)
        K_appro = np.dot(PHI1.T, PHI2)

        # Compute accuracy.
        actual = np.linalg.norm(np.dot(K_appro.T, K_exact), 2)

        self.assertAlmostEquals(actual, self.acc)


if __name__ == '__main__':
    main(verbosity=2)
