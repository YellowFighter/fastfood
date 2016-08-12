#!/usr/local/bin/python3
from os.path import join as pjoin
from time import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.linear_model import ElasticNet

from fastfood import FastfoodPara, FastfoodForKernel

# np.random.seed(0)  # For reproducibility.

d = 520  # Dimension of input pattern
n = d * 20  # Basis number used for approximation
sgm = 10  # Bandwidth for Gaussian kernel

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


def report_ff_en():
    # Fastfood approximation of Gaussian kernel
    para = FastfoodPara(n, d)
    st = time()
    PHI_train, _ = FastfoodForKernel(trainData, para, sgm)
    elapsed_ff_kern_train = time() - st
    st = time()
    PHI_valid, _ = FastfoodForKernel(validationData, para, sgm)
    elapsed_ff_kern_valid = time() - st

    # Train elastic net on projected training data
    en = ElasticNet()
    st = time()
    en.fit(PHI_train.T, trainLabels)
    elapsed_en_fit = time() - st

    # Predict labels for projected validation data
    st = time()
    y_pred = en.predict(PHI_valid.T)
    elapsed_en_pred = time() - st

    # Report performance
    mse_proj = metrics.mean_squared_error(validationLabels, y_pred)
    # print("For projected data, MSE = {:0.4g}.".format(mse_proj))

    return mse_proj, elapsed_en_fit, elapsed_ff_kern_train


def report_orig_en():
    # Train elastic net on original training data
    en = ElasticNet()
    st = time()
    en.fit(trainData.T, trainLabels)
    elapsed_en_fit = time() - st

    # Predict labels for original validation data
    st = time()
    y_pred = en.predict(validationData.T)
    elapsed_en_pred = time() - st

    # Report performance
    mse_orig = metrics.mean_squared_error(validationLabels, y_pred)
    return mse_orig, elapsed_en_fit, 0.


def report_diff():
    orig = report_orig_en()
    proj = report_ff_en()
    return zip(orig, proj)


def main():
    ntrials = 5
    mses = np.zeros(shape=(ntrials, 2))
    times = np.zeros(shape=(ntrials, 3))

    for ii in range(ntrials):
        print('Starting iter {}'.format(ii))
        mse, en_fit, ffkern = report_diff()
        mses[ii, :] = np.array(mse)
        times[ii, :] = np.array(en_fit + (ffkern[1],))

    return mses, times


if __name__ == '__main__':
    mses, times = main()

    # Save results.
    data = np.hstack(
        (np.arange(mses.shape[0]).reshape(mses.shape[0], 1), mses, times))
    header = ["trial", "MSE_orig", "MSE_proj", "ENfit_orig", "ENfit_proj",
              "FFkern_proj"]
    df = pd.DataFrame(data, columns=header)
    df.to_csv('report_en.csv')

    # Visualize results.
    fig = plt.figure()
    plt.plot(df["trial"], df["MSE_orig"], 'ro', df["MSE_proj"], 'go')
    plt.show()
    plt.savefig('report_en.png')
