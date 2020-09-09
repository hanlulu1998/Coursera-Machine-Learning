import numpy as np
import matplotlib.pyplot as plt


def plotDataPoints(X, idx, K=None):
    plt.scatter(X[:, 0], X[:, 1], s=15, c=idx, cmap='hsv')
    return


def drawLine(p1, p2, *args, **kwargs):
    plt.plot([p1[0], p2[0]], [p1[1], p2[1]], *args, **kwargs)
    return


def plotProgresskMeans(X, centroids, previous, idx, K, i):
    plotDataPoints(X, idx, K)
    plt.plot(centroids[:, 0], centroids[:, 1], 'x', markersize=10, linewidth=3)
    for j in range(centroids.shape[0]):
        drawLine(centroids[j, :], previous[j, :])
    plt.title('Iteration number {}'.format(i))
    return


def findClosestCentroids(X, centroids):
    K = centroids.shape[0]
    tmp = np.zeros((X.shape[0], K))
    for i in range(K):
        tmp[:, i] = np.sum((X - centroids[i, :]) ** 2, axis=1)
    idx = np.argmin(tmp, axis=1)

    return idx


def computeCentroids(X, idx, K):
    n = X.shape[1]
    centroids = np.zeros((K, n))
    for i in range(K):
        if sum(idx == i) != 0:
            centroids[i, :] = np.mean(X[np.argwhere(idx == i)[:, 0], :], axis=0)
        else:
            centroids[i, :] = np.zeros((1, n))
    return centroids


def runkMeans(X, initial_centroids, max_iters, plot_progress=False):
    if plot_progress:
        plt.figure()
    m = X.shape[0]
    K = initial_centroids.shape[0]
    centroids = initial_centroids
    previous_centroids = centroids
    idx = np.zeros((m, 1))
    for i in range(max_iters):
        print('K-Means iteration %d/%d...\n' % (i, max_iters))
        idx = findClosestCentroids(X, centroids)
        if plot_progress:
            plotProgresskMeans(X, centroids, previous_centroids, idx, K, i)
            previous_centroids = centroids
        centroids = computeCentroids(X, idx, K)
    if plot_progress:
        plt.show()

    return centroids, idx


def kMeansInitCentroids(X, K):
    randidx = np.random.choice(X.shape[0], K)
    centroids = X[randidx, :]
    return centroids
