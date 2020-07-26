'''
@Descripttion: 
@version: 
@Author: Han Lulu
@Date: 2020-07-22 20:55:15
@LastEditors: Han Lulu
@LastEditTime: 2020-07-23 20:57:40
'''
import matplotlib.pyplot as plt
import numpy as np


def warmUpExercise():
    # TODO Return the 5x5 identity matrix
    return np.eye(5)


def plotData(X, y):
    # TODO plotData Plots the data points x and y into a new figure
    # plotData(x,y) plots the data points and gives the figure axes labels of
    # population and profit.
    plt.plot(X, y, 'rx', markersize=10)
    plt.xlabel('Population of City in 10,000s')
    plt.ylabel('Profit in $10,000s')


def computeCost(X, y, theta):
    # TODO computeCost Compute cost for linear regression
    # J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
    # parameter for linear regression to fit the data points in X and y
    m = len(X)
    h = np.dot(X, theta)
    temp = np.power((h - y), 2)
    J = 1 / (2 * m) * np.sum(temp)
    return J


def gradientDescent(X, y, theta, alpha, iterations):
    # TODO GRADIENTDESCENT Performs gradient descent to learn theta
    # theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by
    # taking num_iters gradient steps with learning rate alpha
    m = len(y)
    J_history = []
    for i in range(iterations):
        h = np.dot(X, theta)
        temp = h - y
        temp0 = theta[0] - alpha * (1 / m) * sum(temp)
        temp1 = theta[1] - alpha * (1 / m) * sum(temp * X[:, 1])
        theta = [temp0, temp1]

        J_history.append(computeCost(X, y, theta))

    return theta, J_history
