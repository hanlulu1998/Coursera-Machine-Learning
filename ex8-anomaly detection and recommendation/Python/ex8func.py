import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt
import math


def estimateGaussian(X):
    mu = np.mean(X, axis=0)
    sigma2 = np.mean((X - mu) ** 2, axis=0)
    return mu, sigma2


def multivariateGaussian(X, mu, sigma2):
    k = len(mu)
    if (sigma2[:, None].shape[0] == 1) or (sigma2[:, None].shape[1] == 1):
        sigma2 = np.diag(sigma2)
    X = X - mu
    p = (2 * math.pi) ** (-k / 2) * np.linalg.det(sigma2) ** (-0.5) * np.exp(
        -0.5 * np.sum((X.dot(np.linalg.pinv(sigma2))) * X, axis=1))
    return p


def visualizeFit(X, mu, sigma2):
    X1, X2 = np.meshgrid(np.arange(0, 36, 0.5), np.arange(0, 36, 0.5))
    Z = multivariateGaussian(np.hstack((X1.reshape((-1, 1), order='F'), X2.reshape((-1, 1), order='F'))), mu,
                             sigma2).reshape(X1.shape, order='F')
    plt.plot(X[:, 0], X[:, 1], 'bx')
    lev = np.power(10, np.arange(-20, 1, 3).astype(np.float))
    print(lev)
    plt.contour(X1, X2, Z, levels=lev)

    return


def selectThreshold(yval, pval):
    bestEpsilon = 0
    bestF1 = 0
    F1 = 0
    stepsize = (max(pval) - min(pval)) / 1000
    for epsilon in np.arange(min(pval), max(pval) + stepsize, stepsize):
        predict = (pval < epsilon)
        Tp = sum((predict == 1) & (yval == 1))
        Fp = sum((predict == 1) & (yval == 0))
        Fn = sum((predict == 0) & (yval == 1))
        precision = Tp / (Tp + Fp)
        recall = Tp / (Tp + Fn)
        F1 = 2 * precision * recall / (precision + recall)
        if F1 > bestF1:
            bestF1 = F1
            bestEpsilon = epsilon
    return bestEpsilon,bestF1
