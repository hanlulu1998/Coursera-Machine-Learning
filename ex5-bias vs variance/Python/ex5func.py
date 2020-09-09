import numpy as np
import copy
import scipy.optimize as opt
import matplotlib.pyplot as plt


def linearRegCostFunction(X, y, theta, lamb):
    m = len(y)
    h = X.dot(theta)
    t = copy.deepcopy(theta)
    t[0] = 0
    J = 1 / (2 * m) * sum((h - y) ** 2) + lamb / (2 * m) * sum(t ** 2)
    grad = 1 / m * X.T.dot(h - y) + lamb / m * t

    return J, grad


def trainLinearReg(X, y, lamb):
    initial_theta = np.random.uniform(-0.12, 0.12, (X.shape[1],))
    costfunc = lambda theta, X, y, lamb: linearRegCostFunction(X, y, theta, lamb)
    result = opt.minimize(fun=costfunc, x0=initial_theta, args=(X, y, lamb), method='tnc', jac=True)
    return result


def learningCurve(X, y, Xval, yval, lamb):
    m = X.shape[0]
    error_train = np.zeros((m, 1))
    error_val = np.zeros((m, 1))
    for i in np.arange(m):
        theta = trainLinearReg(X[0:i + 1, :], y[0:i + 1], lamb).x
        error_train[i], _ = linearRegCostFunction(X[0: i + 1, :], y[0: i + 1], theta, lamb)
        error_val[i], _ = linearRegCostFunction(Xval, yval, theta, lamb)
    return error_train, error_val


def polyFeatures(X, p):
    X_poly = np.zeros((X.size, p))
    for i in range(p):
        X_poly[:, i] = X[:, 0] ** (i + 1)
    return X_poly


def featureNormalize(X):
    mu = np.mean(X, axis=0)
    X_norm = X - mu
    sigma = np.std(X_norm, axis=0)
    X_norm = X_norm / sigma
    return X_norm, mu, sigma


def plotFit(min_x, max_x, mu, sigma, theta, p):
    x = np.arange(min_x - 15, max_x + 25, 0.05)[:, None]
    X_poly = polyFeatures(x, p)
    X_poly = (X_poly - mu) / sigma
    X_poly = np.hstack((np.ones((x.shape[0], 1)), X_poly))
    plt.plot(x, X_poly.dot(theta), '--', linewidth=2)
    return


def validationCurve(X, y, Xval, yval):
    lambda_vec = np.array([0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10])
    m = len(lambda_vec)
    error_train = np.zeros((m, 1))
    error_val = np.zeros((m, 1))
    for i in range(m):
        lamb = lambda_vec[i]
        theta = trainLinearReg(X, y, lamb).x
        error_train[i], _ = linearRegCostFunction(X, y, theta, lamb)
        error_val[i], _ = linearRegCostFunction(Xval, yval, theta, lamb)
    return lambda_vec, error_train, error_val
