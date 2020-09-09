import numpy as np
import copy
import scipy.optimize as opt
import matplotlib.pyplot as plt


def featureNormalize(X):
    mu = np.mean(X, axis=0)
    X_norm = X - mu
    sigma = np.std(X_norm, axis=0, ddof=1)
    X_norm = X_norm / sigma
    return X_norm, mu, sigma


def pca(X):
    m = X.shape[0]
    sigma = 1 / m * (X.T.dot(X))
    U, S, _ = np.linalg.svd(sigma)
    return U, S


def projectData(X, U, K):
    Z = X.dot(U[:, 0:K])
    return Z


def recoverData(Z, U, K):
    X_rec = Z.dot(U[:, 0:K].T)
    return X_rec


def displayData(X):
    BATCH_SIZE = X.shape[0]
    n = X.shape[1]
    SHOW_ROWS = np.floor(np.sqrt(BATCH_SIZE)).astype(dtype=int)
    SHOW_COLS = np.ceil(BATCH_SIZE / SHOW_ROWS).astype(dtype=int)
    example_width = np.round(np.sqrt(n)).astype(dtype=int)
    example_height = (n / example_width).astype(dtype=int)
    for i in range(BATCH_SIZE):
        plt_idx = i + 1
        plt.subplot(SHOW_ROWS, SHOW_COLS, plt_idx)
        plt.imshow(X[i].reshape((example_width, example_height), order='F'))
        plt.axis('off')
    return
