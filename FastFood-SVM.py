
# coding: utf-8

# #Fast Food SVM
# Application of the Fast Food kernel expansion algorithm to SVMs.
# 
# Code ported from MATLAB implementation:
#  * Ji Zhao, Deyu Meng. FastMMD: Ensemble of Circular Discrepancy for Efficient Two-Sample Test. Neural Computation, 2015. 

# In[10]:

from __future__ import division, print_function
get_ipython().magic(u'matplotlib inline')
import matplotlib
import matplotlib.pyplot as plt
#import pylab as pl
import theano
import theano.tensor as T
import numpy as np
from numpy.linalg import norm
import pandas as pd
from sklearn import svm, metrics, datasets
from scipy.linalg import hadamard
from scipy.special import gammaincinv
from functools import partial
from collections import namedtuple
import ipdb

VERBOSE = True  # increases verbosity of some outputs

d = 64 # dimension of input pattern
n = d*20 # basis number used for approximation
sgm = 10 # bandwidth for Gaussian kernel

# random number generator
rng = np.random.RandomState(None)

# load data
digits = datasets.load_digits(2)
N, d = digits.data.shape
X1 = digits.data
print('Shape of X1:',X1.shape)
y = digits.target
print('Shape of y:',y.shape)

np.savetxt('digits-2.csv',X1,delimiter=',')
print("Wrote digits to digits-2.csv")

# pad with zeros so d is nearest power of 2
# N = X.shape[0]
# l = X.shape[1]
# print('Original shape of X: n =',N,', d =',l)
# d = int(2 ** np.ceil(np.log2(l)))
# if d != l:  # pad only if needed
#     print('Padding input from d =',l,'to d =',d)
#     X = np.pad(X,((0,d-l),(0,0)),mode='constant',constant_values=0)
    
# convert to shared tensors
#X = theano.shared(X,name='X',borrow=True)
#y = theano.shared(y,name='y',borrow=True)


# In[11]:

FFPara = namedtuple('FFPara', 'B G PI S')
def fastfood_params(n,d):
    d0 = d
    n0 = n
    l = int(np.ceil(np.log2(d)))
    d = 2**l
    k = int(np.ceil(n/d))
    n = d*k
    print('d0 =',d0,', d =',d)
    print('n0 =',n,', n =',n)
    
    B = []
    G = []
    PI = []
    S = []
    for ii in xrange(k):
        B_ii  = rng.choice([-1,1],size=d)
        G_ii  = rng.normal(size=d)
        PI_ii = rng.permutation(d)
        
        B.append(B_ii)
        G.append(G_ii)
        PI.append(PI_ii)
        
        p1 = rng.uniform(size=d)
        p2 = d/2
#        print('p1 =',p1,'; p2 =',p2)
        T = gammaincinv(p2,p1)
#        print('T1 =',T)
        T = (T*2) ** (1/2)
#        print('T2 =',T)
        s_i = T * norm(G,'fro')**(-1)
#        print('s_i =', s_i)
        S_ii = s_i
        S.append(S_ii)
    
    S1 = np.zeros(n)
    for ii in xrange(k):
        S1[ii*d:(ii+1)*d] = S[ii]
    
#    print('Shape of B:',len(B),', B[0]:',B[0].shape)
    
    return FFPara(B, G, PI, S1)
print('Ready to generate fastfood params')


# In[20]:

def bit_reverse_traverse(a):
    # (c) 2014 Ryan Compton
    # ryancompton.net/2014/06/05/bit-reversal-permutation-in-python/
    n = a.shape[0]
    assert(not n&(n-1) ) # assert that n is a power of 2
    if n == 1:
        yield a[0]
    else:
        even_index = np.arange(int(n/2))*2
        odd_index = np.arange(int(n/2))*2 + 1
        for even in bit_reverse_traverse(a[even_index]):
            yield even
        for odd in bit_reverse_traverse(a[odd_index]):
            yield odd

def get_bit_reversed_list(l):
    # (c) 2014 Ryan Compton
    # ryancompton.net/2014/06/05/bit-reversal-permutation-in-python/
    n = len(l)
#    print('n=',n)
    indexs = np.arange(n)
    b = []
    for i in bit_reverse_traverse(indexs):
        b.append(l[i])
    return b

def FWHT(X):
    # Fast Walsh-Hadamard Transform for 1D signals
    # of length n=2^M only (non error-proof for now)
    x=get_bit_reversed_list(X)
    x=np.array(x)
    N=len(X)
 
    for i in range(0,N,2):
        x[i]=x[i]+x[i+1]
        x[i+1]=x[i]-2*x[i+1]
 
    L=1
    y=np.zeros_like(x)
    for n in range(2,int(np.log2(N))+1):
        M=2**L
        J=0; K=0
        while(K<N):
            for j in range(J,J+M,2):
                y[K]   = x[j]   + x[j+M]
                y[K+1] = x[j]   - x[j+M]
                y[K+2] = x[j+1] + x[j+1+M]
                y[K+3] = x[j+1] - x[j+1+M]
                K=K+4
            J=J+2*M
        x=y.copy()
        L=L+1
 
    y=x/float(N)
    
    return y

def fastfood_forkernel(X,para,sgm):
    d0, m = X.shape
#    print('d0 =',d0,', m =',m)
    l = int(np.ceil(np.log2(d0)))
    d = 2**l
    if d == d0:
        XX = X
    else:
        XX = np.zeros((d,m))
        XX[0:d0,:] = X
#    print('d=',d,',m=',m)
#         print('d =',d)
#    print('Shape of XX:',XX.shape)
    
    k = len(para.B)
#     print("k=",k)
    n = d*k
    tht = np.zeros((n,m))
#     print('(n,m) =',(n,m))
    for ii in xrange(k):
#         print("ii=",ii)
        B = para.B[ii]
        G = para.G[ii]
        PI = para.PI[ii]
#        XX = np.dot(B,XX)
        XX = np.dot(np.diag(B),XX)
#         print(XX.shape)
        T = FWHT(XX)
        T_m = pd.read_csv('/Users/kellanfluette/dev/fastfood/samples/matlab_fastfood/Fastfood/fwht-xx-'+str(ii+1)+'.csv',header=None)
        print('T =',T,', T_m =',T_m)
#         print('PI =',PI)
        T = T[PI,:]
#         print('T =',T)
#        print('T.shape:',T.shape,',(G*d).shape:',(G*d).shape)
        T = np.dot(np.diag(G*d),T)
#        print('T.shape:',T.shape)
        T = FWHT(T)
#        print('T.shape:',T.shape)
        idx1 = ii*d
        idx2=(ii+1)*d
#        print('idx1=',idx1,',idx2=',idx2)
        tht[idx1:idx2,:] = T
        
    S = para.S
    print('tht.shape:',tht.shape)
    #ipdb.set_trace()
#     tht = np.dot(tht,np.diag(S*np.sqrt(d)))
    #tht = np.dot(S*np.sqrt(d),tht)  # TODO: make dot product ... ?
    tht = (S*np.sqrt(d)*tht.T).T

    T = tht/sgm
    phi = np.concatenate([np.cos(T),np.sin(T)],axis=1)
    phi = 1.0/np.sqrt(n) * phi
    return phi,tht


# In[21]:

def ff_kernel(para,sgm,x1,x2):
    # TODO use the precomputed Gram matrix rather than calculating K(x,y) individually
    X = np.hstack((x1,x2))
    phi, tht = fastfood_forkernel(X,para,sgm)
    K_appro = np.dot(phi[0].T,phi[1])
    return K_appro


# In[22]:

para = fastfood_params(n,d)
#print('Fastfood params:',params)

def compute_kernel_matrix(X,phi):
    d0, m = X.shape
    l = int(np.ceil(np.log2(d0)))
    d = 2**l
    if d == d0:
        XX = X
    else:
        XX = np.zeros((d,m))
        XX[0:d0,:] = X
#     print('XX.shape:',XX.shape)

    K = np.zeros((N,N),dtype=np.float64)
    for i in xrange(N):
        for j in xrange(N):
#            phi1 = np.dot(phi[i],XX[:,i])
#            phi2 = np.dot(phi[j],XX[:,j])
            phi_i = phi[i]
            phi_j = phi[j]
            K_ij = np.dot(phi_i,phi_j)
            K[i,j] = K_ij
    return K

X1T = X1.T
# print('X1.shape:',X1.shape)
phi1,tht1 = fastfood_forkernel(X1T,para,sgm)
#print(phi1[1:3,:])
# print('phi1.shape:',phi1.shape)
phi1 = pd.read_csv('/Users/kellanfluette/dev/fastfood/samples/matlab_fastfood/Fastfood/phi1.csv',header=None)
K_appro = compute_kernel_matrix(X1T,phi1)
# print('K_appro =',K_appro)
# print('K_appro.shape:',K_appro.shape)


# In[16]:

C = 1.0
gamma = 0.001

# we create an instance of SVM and fit out data.
#my_kernel = lambda x1,x2: ff_kernel(para,sgm,x1,x2)
clf = svm.SVC(kernel='precomputed',gamma=gamma,C=C,random_state=rng)
clf.fit(K_appro,y)
y_pred = clf.predict(K_appro)
print('ff-kSVM metrics:\n',metrics.classification_report(y,y_pred))

clf2 = svm.SVC(kernel='rbf',C=C,gamma=gamma,random_state=rng)
#print('X1.shape:',X1.shape,', y.shape:',y.shape)
clf2.fit(X1,y)
y_pred = clf2.predict(X1)
print('kSVM(rbf) metrics:\n',metrics.classification_report(y,y_pred))


# In[10]:

d


# In[24]:

tht=np.array([[1,2,3],[4,5,6],[7,8,9]])
S=np.array([-1,1,-1])
d=3
(S*np.sqrt(d)*tht.T).T


# In[21]:

S.shape


# In[11]:

tht


# In[ ]:



