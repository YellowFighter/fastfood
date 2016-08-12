from os import path
import numpy as np
import numpy.random as rand
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_mldata
# from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import enet_path

__doc__ = "See newcomparison.m"

l1_ratio = 0.5
k_fold = 10
test_frac = 0.5
data_root = path.expanduser('~/data')

# Load MNIST data
mnist = fetch_mldata('MNIST original', data_home=data_root)
# N, D = mnist.data.shape
X = mnist.data
y = mnist.target
# mnist.data.shape == (70000, 784)
# mnist.target.shape == (70000,)
# np.unique(mnist.target) == array([0., 1., 2., 3., 4., 5., 6., 7., 8., 9.])

# random_state=0 for reproducibility
# en = ElasticNetCV(cv=k_fold, n_jobs=-1, random_state=0)
# en.fit(X, y)


# Compute paths
# alphas_enet, coefs_enet, _ = enet_path(X, y)

# Display results
plt.figure()
ax = plt.gca()
l2 = plt.plot(-np.log10(alphas_enet), coefs_enet.T, linestyle='--')
plt.xlabel('-Log(alpha)')
plt.ylabel('coefficients')
plt.title('Elastic-Net Paths')
plt.axis('tight')
plt.savefig('comparison2.svg')
