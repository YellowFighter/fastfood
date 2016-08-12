from os import path
import numpy as np
import numpy.random as rand
from sklearn.datasets import fetch_mldata

from util import train_test_inxs

__doc__ = """See comparison.m.
             This script is to run EN and FFEN on linear and nonlinear datasets
             for various values of N, d."""

n_iter = 10
test_frac = 0.5
data_root = path.expanduser('~/data')

# Load MNIST data
mnist = fetch_mldata('MNIST original', data_home=data_root)
N, D = mnist.data.shape
# mnist.data.shape == (70000, 784)
# mnist.target.shape == (70000,)
# np.unique(mnist.target) == array([0., 1., 2., 3., 4., 5., 6., 7., 8., 9.])

# Partition into train/test
train_inxs, test_inxs = train_test_inxs(N, test_frac)
train_data = mnist.data[train_inxs]
test_data = mnist.data[test_inxs]
train_targets = mnist.targets[train_inxs]
test_targets = mnist.target[test_inxs]
