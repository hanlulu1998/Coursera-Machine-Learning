import matplotlib.pyplot as plt
import numpy as np
from ex2func import sigmoid, plotData


def mapFeature(X1, X2):
    # MAPFEATURE Feature mapping function to polynomial features
    #
    #   MAPFEATURE(X1, X2) maps the two input features
    #   to quadratic features used in the regularization exercise.
    #
    #   Returns a new feature array with more features, comprising of
    #   X1, X2, X1.^2, X2.^2, X1*X2, X1*X2.^2, etc..
    #
    #   Inputs X1, X2 must be the same size
    degree = 6
    out = np.ones(X1.shape)[:, None]
    for i in range(1, degree + 1):
        for j in range(i + 1):
            temp = (X1 ** (i - j)) * (X2 ** j)
            out = np.hstack((out, temp[:, None]))
    return out


def costFunctionReg(theta, X, y, l):
    # COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
    #   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
    #   theta as the parameter for regularized logistic regression and the
    #   gradient of the cost w.r.t. to the parameters.
    m = len(y)
    z = X.dot(theta)
    h = sigmoid(z)
    J = 1 / m * sum(-y * np.log(h) - (1 - y) * np.log(1 - h)) + l / (2 * m) * sum(theta ** 2)
    grad = 1 / m * sum((h - y)[:, None] * X)
    grad[1:-1] = grad[1:-1] + l / m * theta[1:-1]
    return J, grad


# serve scipy.optimize.minimize function
def costfunc(theta, X, y, l):
    return costFunctionReg(theta, X, y, l)[0]


def gradfunc(theta, X, y, l):
    return costFunctionReg(theta, X, y, l)[1]


def plotDecisionBoundary(theta, X, y):
    # PLOTDECISIONBOUNDARY Plots the data points X and y into a new figure with
    # the decision boundary defined by theta
    plotData(X[:, [1, 2]], y)
    if X.shape[1] <= 3:
        plot_x = np.array([np.min(X[:, 1]) - 2, np.max(X[:, 1]) + 2])
        plot_y = (-1 / theta[2]) * (theta[1] * plot_x + theta[0])
        plt.plot(plot_x, plot_y)
        plt.legend(['Admitted', 'Not admitted', 'Decision Boundary'])
        plt.axis([30, 100, 30, 100])
    else:
        u = np.linspace(-1, 1.5, 50)
        v = np.linspace(-1, 1.5, 50)
        z = np.zeros((len(u), len(v)))
        for i in range(len(u)):
            for j in range(len(v)):
                z[i, j] = mapFeature(u[i].reshape(1, ), v[j].reshape(1, )).dot(theta)
        print(z.shape)
        plt.contour(u, v, z, levels=0, linewidth=2)
