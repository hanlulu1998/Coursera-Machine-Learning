import numpy as np
from ex3func import sigmoid


def predict(Theta1, Theta2, X):
    m = X.shape[0]
    X = np.hstack((np.ones((m, 1)), X))
    h1 = sigmoid(X.dot(Theta1.T))
    h1 = np.hstack((np.ones((m, 1)), h1))
    h2 = sigmoid(h1.dot(Theta2.T))
    p = np.argmax(h2, axis=1)+1
    return p
