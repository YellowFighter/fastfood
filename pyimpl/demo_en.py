from os.path import join as pjoin
from time import time
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn import metrics
from fastfood import FastfoodPara, FastfoodForKernel

d = 520  # Dimension of input pattern
n = d * 20  # Basis number used for approximation
sgm = 10  # Bandwidth for Gaussian kernel

np.random.seed(0)  # For reproducibility.

# Load data
data_dir = '/Users/kellanfluette/dev/fastfood/data/UJIndoorLoc/'
trainData = pd.read_csv(pjoin(data_dir, 'trainingData.csv'))
lat_long = trainData[["LATITUDE", "LONGITUDE"]].as_matrix()
trainLabels = (lat_long[:, 0] + 90.0) * 180.0 + lat_long[:, 1]
trainData = trainData.as_matrix()[:, :-9].T
validationData = pd.read_csv(pjoin(data_dir, 'validationData.csv'))
lat_long = validationData[["LATITUDE", "LONGITUDE"]].as_matrix()
validationLabels = (lat_long[:, 0] + 90.0) * 180.0 + lat_long[:, 1]
validationData = validationData.as_matrix()[:, :-9].T

# Fastfood approximation of Gaussian kernel
para = FastfoodPara(n, d)
st = time()
PHI_train, _ = FastfoodForKernel(trainData, para, sgm)
elapsed_ff_kern_train = time() - st
print("Took {:0.4g}s to compute training Fastfood expansion.".format(elapsed_ff_kern_train))
st = time()
PHI_valid, _ = FastfoodForKernel(validationData, para, sgm)
elapsed_ff_kern_valid = time() - st
print("Took {:0.4g}s to compute validation Fastfood expansion.".format(elapsed_ff_kern_valid))

# Train elastic net on projected training data
en = ElasticNet()
st = time()
en.fit(PHI_train.T, trainLabels)
elapsed_en_fit = time() - st
print("Took {:0.4g}s to fit elastic net on projected training data.".format(elapsed_en_fit))

# Predict labels for projected validation data
st = time()
y_pred = en.predict(PHI_valid.T)
elapsed_en_pred = time() - st
print("Took {:0.4g}s to predict on projected validation data.".format(elapsed_en_pred))

# Report performance
mse_proj = metrics.mean_squared_error(validationLabels, y_pred)
print("For projected data, MSE = {:0.4g}.".format(mse_proj))

# Train elastic net on original training data
en = ElasticNet()
st = time()
en.fit(trainData.T, trainLabels)
elapsed_en_fit = time() - st
print("Took {:0.4g}s to fit elastic net on original training data.".format(elapsed_en_fit))

# Predict labels for original validation data
st = time()
y_pred = en.predict(validationData.T)
elapsed_en_pred = time() - st
print("Took {:0.4g}s to predict on projected validation data.".format(elapsed_en_pred))

# Report performance
mse_orig = metrics.mean_squared_error(validationLabels, y_pred)
print("For original data, MSE = {:0.4g}.".format(mse_orig))

print("MSE decrease by projecting = {:0.4g}.".format(mse_orig - mse_proj))
