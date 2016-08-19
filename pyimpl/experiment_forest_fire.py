from __future__ import print_function

from os import path

import random
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rand
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.datasets import fetch_mldata
from sklearn.linear_model import ElasticNet
from sklearn.metrics import (
    mean_squared_error, explained_variance_score, r2_score)
from sklearn.preprocessing import LabelEncoder
from fastfood import FastfoodPara, FastfoodForKernel, pad_to_power_of_2

rand.seed(0)
random.seed(0)

l1_ratio = 0.5
test_frac = 0.5

# Load Forest Fires data
data = pd.read_csv('../data_sets/forest_fires/forestfires.csv')
le_mo = LabelEncoder()
data[['month']] = le_mo.fit_transform(data[['month']])
le_day = LabelEncoder()
data[['day']] = le_day.fit_transform(data[['day']])
X = data.ix[:, data.columns != 'area'].as_matrix()
y = data[["area"]].as_matrix()

# Split into train/test_frac
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_frac, random_state=0)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

# Construct and fit EN model
en = ElasticNet(random_state=0)
en.fit(X_train, y_train)

# Evaluate performance
y_pred = en.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
exp_var = explained_variance_score(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print('** EN **')
print('MSE: %.4f' % (mse,))
print('Explained variance: %.4f' % (exp_var,))
print('R2: %.4f' % (r2,))

# Construct and fit FFEN model
X_train = pad_to_power_of_2(X_train.T)
X_test = pad_to_power_of_2(X_test.T)
print(X_train.shape)
print(X_test.shape)
d = X_train.shape[0]  # Dimension of input pattern
n = d * 20  # Basis number used for approximation
sgm = 10  # Bandwidth for Gaussian kernel
para = FastfoodPara(n, d)
PHI_train, _ = FastfoodForKernel(X_train, para, sgm)
PHI_test, _ = FastfoodForKernel(X_test, para, sgm)

en = ElasticNet(random_state=0)
en.fit(PHI_train.T, y_train)

# Evaluate performance
y_pred = en.predict(PHI_test.T)
mse = mean_squared_error(y_test, y_pred)
exp_var = explained_variance_score(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print('** FFEN **')
print('MSE: %.4f' % (mse,))
print('Explained variance: %.4f' % (exp_var,))
print('R2: %.4f' % (r2,))
