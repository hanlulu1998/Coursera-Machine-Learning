'''
@Descripttion: 
@version: 
@Author: Han Lulu
@Date: 2020-07-22 20:55:15
@LastEditors: Han Lulu
@LastEditTime: 2020-07-23 20:57:40
'''
import numpy as np


def featureNormalize(X):
    # TODO FEATURENORMALIZE Normalizes the features in X
    #   FEATURENORMALIZE(X) returns a normalized version of X where
    #   the mean value of each feature is 0 and the standard deviation
    #   is 1. This is often a good preprocessing step to do when
    #   working with learning algorithms.
    X_norm = X
    mu = np.mean(X_norm)
    sigma = np.std(X_norm)
    X_norm = (X_norm - mu) / sigma
    return X_norm, mu, sigma


def computeCostMulti(X, y, theta):
    # TODO computeCost Compute cost for linear regression
    # J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
    # parameter for linear regression to fit the data points in X and y
    m = len(X)
    h = np.dot(X, theta)
    temp = np.power((h - y), 2)
    J = 1 / (2 * m) * np.sum(temp)
    return J


def gradientDescentMulti(X, y, theta, alpha, num_iters):
    # TODO GRADIENTDESCENT Performs gradient descent to learn theta
    # theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by
    # taking num_iters gradient steps with learning rate alpha
    m = len(y)
    J_history = []
    for i in range(num_iters):
        h = np.dot(X, theta)
        mid = h - y
        temp = alpha * (1 / m) * sum(mid[:, None] * X)
        temp = theta - temp
        theta = temp

        J_history.append(computeCostMulti(X, y, theta))

    return theta, J_history


def normalEqn(X, y):
    # NORMALEQN Computes the closed-form solution to linear regression
    #   NORMALEQN(X,y) computes the closed-form solution to linear
    #   regression using the normal equations.
    theta = np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(y)
    return theta
