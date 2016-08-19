from __future__ import print_function
from os import path
import numpy as np
import numpy.random as rand
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_mldata
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.cross_validation import train_test_split

__doc__ = "See newcomparison.m"

l1_ratio = 0.5
k_fold = 10
test_frac = 0.5
data_root = path.expanduser('~/data')

# Load MNIST data
mnist = fetch_mldata('MNIST original', data_home=data_root)
X = mnist.data
y = mnist.target

# Split into train/test_frac
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_frac, random_state=0)

# Construct and fit model
en = ElasticNetCV(cv=k_fold, n_jobs=-1, random_state=0)
en.fit(X_train, y_train)

# Evaluate performance
y_pred = np.round(en.predict(X_test))
conf_mat = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)

print(acc)
