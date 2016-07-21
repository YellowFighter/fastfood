import numpy as np
from fastfood import FastfoodPara, FastfoodForKernel

d = 64  # Dimension of input pattern
n = d * 20  # Basis number used for approximation
sgm = 10  # Bandwidth for Gaussian kernel
N = 10  # Number of sample input patterns to generate

np.random.seed(0)  # For reproducibility.

# Generate random data.
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
acc = np.linalg.norm(np.dot(K_appro.T, K_exact), 2)

# Report accuracy.
print("Difference between Fastfood approximation and exact calculation of Gaussian kernel:")
loss_pct = np.asscalar((1.0 - acc) * 100.0)
print("{:0.2f}%".format(loss_pct))
