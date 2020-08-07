import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt


def sigmoid(z):
    # SIGMOID Compute sigmoid function
    # g = SIGMOID(z) computes the sigmoid of z.
    g = 1 / (1 + np.exp(-z))
    return g


def displayData(X):
    BATCH_SIZE = 100
    SHOW_ROWS, SHOW_COLS = 10, 10
    # set some useful parameter
    for i in range(BATCH_SIZE):
        plt_idx = i + 1
        plt.subplot(SHOW_ROWS, SHOW_COLS, plt_idx)
        idx = np.random.choice(a=np.size(X, 0), size=1, replace=False, p=None)
        deal_val = np.mean(np.abs(X[idx]))
        plt_arry = X[idx] / deal_val
        plt.imshow(plt_arry.reshape((20,20),order='F').astype('uint8'), cmap='gray', vmax=1, vmin=-1)
        plt.axis('off')
    return


def lrCostFunction(theta, X, y, l):
    m = len(y)
    z = X.dot(theta)
    h = sigmoid(z)
    J = 1 / m * sum(-y * np.log(h) - (1 - y) * np.log(1 - h)) + l / (2 * m) * sum(theta[1:] ** 2)
    grad = 1 / m * X.T.dot(h - y)
    grad[1:] = grad[1:] + l / m * theta[1:]
    return J, grad


def costfunc(theta, X, y, l):
    return lrCostFunction(theta, X, y, l)[0]


def gradfunc(theta, X, y, l):
    return lrCostFunction(theta, X, y, l)[1]


def oneVsAll(X, y, num_labels, l):
    (m, n) = X.shape
    all_theta = np.zeros((num_labels, n + 1))
    X = np.hstack((np.ones((m, 1)), X))
    initial_theta = np.zeros((n + 1,))
    for i in range(num_labels):
        if i == 0:
            index = 10
        else:
            index = i
        result = opt.minimize(fun=costfunc, x0=initial_theta, args=(X, (y == index).astype(np.int), l), method='TNC',
                              jac=gradfunc)
        theta = result.x
        all_theta[i, :] = theta

    return all_theta


def predictOneVsAll(all_theta, X):
    z = X.dot(all_theta.T)
    h = sigmoid(z)
    p = np.argmax(h, axis=1)
    p[p == 0] = 10
    return p
