#!/usr/bin/env python
from theano import function
import theano.tensor as T
from theano.tensor import shared_randomstreams
import numpy as np
import numpy.random
from scipy.special import gammaincinv
from numpy.linalg import norm

# tensor stand-in for np.random.RandomState
rngT = shared_randomstreams.RandomStreams()
rng = numpy.random.RandomState()

# {{{ Fastfood Params }}}
n, d = T.dscalars('n', 'd')
# transform dimensions to be a power of 2
d0, n0 = d, n
l = T.ceil(T.log2(d))  # TODO cast to int
d = 2**l
k = T.ceil(n/d)  # TODO cast to int
n = d*k
# generate parameter 'matrices'
B = rng.choice([-1, 1], size=(k, d))
G = rng.normal(size=(k, d), dtype=np.float64)
PI = np.array([rng.permutation(d) for _ in xrange(k)]).T
S = np.empty((k*d, 1), dtype=np.float64)
# generate scaling matrix, S
for i in xrange(k):
    for j in xrange(d):
        p1 = rng.uniform(size=d)
        p2 = d/2
        Tmp = gammaincinv(p2, p1)
        Tmp = T.sqrt(2*Tmp)
        s_ij = Tmp * norm(G, 'fro')**(-1)
        S[i+j] = s_ij
fastfood_params = function([n, d], [B, G, PI, S])


# {{{ Fastfood for kernel }}}
# params
X = T.dmatrix('X')  # COLUMNS are input patterns (d0 dims, m samples)
B = T.dmatrix('B')
G = T.dmatrix('G')
PI = T.dmatrix('PI')
S = T.dmatrix('S')
para = [B, G, PI, S]
sgm = T.dscalar('sgm')
# book keeping
d0, m = X.shape
l = T.ceil(T.log2(d0))  # TODO cast to int
d = 2**l
if d == d0:
    XX = X
else:
    XX = np.zeros((d, m))
    XX[0:d0, :] = X
#
k = len(para.B)
n = d*k
tht = np.zeros((n, m))
XX = B*XX
Tmp = FWHT(XX)
Tmp = Tmp[PI]  # TODO make this indexing work as expected
Tmp = G*d*Tmp
Tmp = FWHT(T)
tht[ii*d:(ii+1)*d, :] = Tmp
fastfood_forkernel = function([X, para, sgm], [phi, tht])
