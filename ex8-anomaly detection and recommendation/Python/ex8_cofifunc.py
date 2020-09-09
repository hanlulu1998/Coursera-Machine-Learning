import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt
import copy
import re


def cofiCostFunc(params, Y, R, num_users, num_movies, num_features, lamb):
    X = np.reshape(params[0:num_movies * num_features], (num_movies, num_features))
    Theta = np.reshape(params[num_movies * num_features:], (num_users, num_features))
    H = X.dot(Theta.T)
    J = 1 / 2 * sum(sum(R * (H - Y) ** 2)) + lamb / 2 * sum(sum(Theta ** 2)) + lamb / 2 * sum(sum(X ** 2))

    X_grad = np.zeros(X.shape)
    Theta_grad = np.zeros(Theta.shape)
    for i in range(num_movies):
        idx = np.argwhere(R[i, :] == 1)[:, 0]
        Theta_tmp = Theta[idx, :]
        Y_tmp = Y[i, idx]
        X_grad[i, :] = (X[i, :].dot(Theta_tmp.T) - Y_tmp).dot(Theta_tmp) + lamb * X[i, :]
    for i in range(num_users):
        idx = np.argwhere(R[:, i] == 1)[:, 0]
        Y_tmp = Y[idx, i]
        X_tmp = X[idx, :]
        Theta_grad[i, :] = (X_tmp.dot(Theta[i, :].T) - Y_tmp).T.dot(X_tmp) + lamb * Theta[i, :]
    grad = np.append(X_grad, Theta_grad)
    return J, grad


def computeNumericalGradient(J, theta):
    numgrad = np.zeros(theta.shape)
    perturb = np.zeros(theta.shape)
    e = 1e-4
    for p in range(len(theta)):
        # Set perturbation vector
        perturb[p] = e
        loss1 = J(theta - perturb)[0]
        loss2 = J(theta + perturb)[0]
        # Compute Numerical Gradient
        numgrad[p] = (loss2 - loss1) / (2 * e)
        perturb[p] = 0
    return numgrad


def checkCostFunction(lamb=0):
    X_t = np.random.rand(4, 3)
    Theta_t = np.random.rand(5, 3)
    Y = X_t.dot(Theta_t.T)
    Y[np.random.rand(Y.shape[0], Y.shape[1]) > 0.5] = 0
    R = np.zeros(Y.shape)
    R[Y != 0] = 1
    X = np.random.randn(X_t.shape[0], X_t.shape[1])
    Theta = np.random.randn(Theta_t.shape[0], Theta_t.shape[1])
    num_users = Y.shape[1]
    num_movies = Y.shape[0]
    num_features = Theta.shape[1]
    # Unroll parameters
    nn_params = np.append(X, Theta)
    costFunc = lambda p: cofiCostFunc(p, Y, R, num_users, num_movies, num_features, lamb)
    _, grad = costFunc(nn_params)
    numgrad = computeNumericalGradient(costFunc, nn_params)
    print('The above two columns you get should be very similar.\n (Left-Your Numerical Gradient, Right-Analytical '
          'Gradient)\n\n')
    diff = np.linalg.norm(numgrad - grad) / np.linalg.norm(numgrad + grad)
    print('If your backpropagation implementation is correct, then \n'
          'the relative difference will be small (less than 1e-9). \n'
          '\nRelative Difference: {} \n'.format(diff))
    return


def loadMovieList():
    movie_list = []
    with open('movie_ids.txt', 'r', encoding='ISO-8859-1') as f:
        movie_content = f.read().split('\n')
    for movie_str in movie_content:
        movie_tmp = movie_str.split(' ')
        movie_tmp = ' '.join(movie_tmp[1:])
        movie_list.append(movie_tmp)
    return movie_list


def normalizeRatings(Y, R):
    m = Y.shape[0]
    Ymean = np.zeros((m, 1))
    Ynorm = np.zeros(Y.shape)
    for i in range(m):
        idx = np.argwhere(R[i, :] == 1)[:, 0]
        Ymean[i] = np.mean(Y[i, idx])
        Ynorm[i, idx] = Y[i, idx] - Ymean[i]
    return Ynorm, Ymean
