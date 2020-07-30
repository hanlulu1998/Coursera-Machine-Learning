import matplotlib.pyplot as plt
import numpy as np


def plotData(X, y):
    # PLOTDATA Plots the data points X and y into a new figure
    # PLOTDATA(x,y) plots the data points with + for the positive examples
    # and o for the negative examples. X is assumed to be a Mx2 matrix.
    pos = np.argwhere(y == 1)[:, 0]
    neg = np.argwhere(y == 0)[:, 0]
    plt.plot(X[pos, 0], X[pos, 1], 'k+', linewidth=2, markersize=7)
    plt.plot(X[neg, 0], X[neg, 1], 'ko', color='y', markersize=7)


def sigmoid(z):
    # SIGMOID Compute sigmoid function
    # g = SIGMOID(z) computes the sigmoid of z.
    g = 1 / (1 + np.exp(-z))
    return g


def costFunction(theta, X, y):
    # COSTFUNCTION Compute cost and gradient for logistic regression
    # J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
    # parameter for logistic regression and the gradient of the cost
    # w.r.t. to the parameters.
    m = len(y)
    z = X.dot(theta)
    h = sigmoid(z)
    J = 1 / m * sum(-y * np.log(h) - (1 - y) * np.log(1 - h))
    grad = 1 / m * sum((h - y)[:, None] * X)
    return J, grad


# serve scipy.optimize.minimize function
def costfunc(theta, X, y):
    return costFunction(theta, X, y)[0]


def gradfunc(theta, X, y):
    return costFunction(theta, X, y)[1]


def predict(theta, X):
    # PREDICT Predict whether the label is 0 or 1 using learned logistic
    # regression parameters theta
    #   p = PREDICT(theta, X) computes the predictions for X using a
    #   threshold at 0.5 (i.e., if sigmoid(theta'*x) >= 0.5, predict 1)
    m = X.shape[0]
    p = np.zeros((m, 1))
    z = X.dot(theta)
    h = sigmoid(z)
    one = np.argwhere(h >= 0.5)[:, 0]
    p[one] = 1
    return p
