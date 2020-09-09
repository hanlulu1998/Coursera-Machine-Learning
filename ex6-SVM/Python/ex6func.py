import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm


def plotData(X, y):
    # PLOTDATA Plots the data points X and y into a new figure
    # PLOTDATA(x,y) plots the data points with + for the positive examples
    # and o for the negative examples. X is assumed to be a Mx2 matrix.
    pos = np.argwhere(y == 1)[:, 0]
    neg = np.argwhere(y == 0)[:, 0]
    plt.plot(X[pos, 0], X[pos, 1], 'k+', linewidth=2, markersize=7)
    plt.plot(X[neg, 0], X[neg, 1], 'ko', color='y', markersize=7)


def visualizeBoundaryLinear(X, y, confidence):
    pos = np.argwhere(y == 1)[:, 0]
    neg = np.argwhere(y == 0)[:, 0]
    plt.plot(X[pos, 0], X[pos, 1], 'w+', linewidth=0.1, markersize=10)
    plt.scatter(X[:, 0], X[:, 1], s=30, c=confidence, cmap='seismic')
    plt.title('SVM (C=1) Decision Confidence')

    return


def gaussianKernel(x1, x2, sigma):
    sim = np.exp(-sum((x1 - x2) ** 2) / (2 * sigma ** 2))
    return sim


def dataset3Params(X, y, Xval, yval):
    num = np.array([0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30])
    max_correct = -100
    for C in num:
        for sigma in num:
            model = svm.SVC(C=C, kernel='rbf', gamma=sigma)
            model.fit(X, y)
            correct = model.score(Xval, yval)
            if max_correct <= correct:
                max_correct = correct
                best_C = C
                best_sigma = sigma
    return best_C, best_sigma
