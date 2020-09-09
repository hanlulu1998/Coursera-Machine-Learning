import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt
import copy


def sigmoid(z):
    # SIGMOID Compute sigmoid function
    # g = SIGMOID(z) computes the sigmoid of z.
    g = 1 / (1 + np.exp(-z))
    return g


def sigmoidGradient(z):
    g = sigmoid(z) * (1 - sigmoid(z))
    return g


def displayData(X):
    BATCH_SIZE = 100
    SHOW_ROWS, SHOW_COLS = 10, 10
    # set some useful parameter H
    for i in range(BATCH_SIZE):
        plt_idx = i + 1
        plt.subplot(SHOW_ROWS, SHOW_COLS, plt_idx)
        idx = np.random.choice(a=np.size(X, 0), size=1, replace=False, p=None)
        deal_val = np.mean(np.abs(X[idx]))
        plt_arry = X[idx] / deal_val
        plt.imshow(plt_arry.reshape((20, 20), order='F').astype('uint8'), cmap='gray', vmax=1, vmin=-1)
        plt.axis('off')
    return


def nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lamb):
    Theta1 = np.reshape(nn_params[0:hidden_layer_size * (input_layer_size + 1)],
                        (hidden_layer_size, (input_layer_size + 1)))
    Theta2 = np.reshape(nn_params[hidden_layer_size * (input_layer_size + 1):],
                        (num_labels, (hidden_layer_size + 1)))
    m = X.shape[0]
    X = np.hstack((np.ones((m, 1)), X))
    a1 = X.T
    a2 = sigmoid(Theta1 @ a1)
    a2 = np.vstack((np.ones((1, m)), a2))
    a3 = sigmoid(Theta2 @ a2)
    h = a3.T
    yi = np.zeros((m, num_labels))
    for i in range(m):
        yi[i, y[i] - 1] = 1
    y = yi
    tmp1 = copy.deepcopy(Theta1)
    tmp2 = copy.deepcopy(Theta2)
    tmp1[:, 0] = 0
    tmp2[:, 0] = 0
    J = (-1 / m) * sum(sum(y * np.log(h) + (1 - y) * np.log(1 - h))) + lamb / (2 * m) * (
            sum(sum(tmp1 ** 2)) + sum(sum(tmp2 ** 2)))

    dTheta1 = np.zeros(Theta1.shape)
    dTheta2 = np.zeros(Theta2.shape)
    for t in range(m):
        a1 = X[t, :].T
        yt = y[t, :].T
        z2 = Theta1 @ a1
        a2 = sigmoid(z2)
        a2 = np.insert(a2, 0, 1)
        z3 = Theta2 @ a2
        a3 = sigmoid(z3)

        dx3 = a3 - yt
        z2 = np.insert(z2, 0, 1)
        dx2 = Theta2.T @ dx3 * sigmoidGradient(z2)
        dx2 = dx2[1:]
        dTheta2 = dTheta2 + dx3[:, None] @ a2[None, :]
        dTheta1 = dTheta1 + dx2[:, None] @ a1[None, :]

    Theta1_grad = 1 / m * dTheta1
    Theta2_grad = 1 / m * dTheta2

    Theta2_grad[:, 1:] = Theta2_grad[:, 1:] + lamb * Theta2[:, 1:] / m
    Theta1_grad[:, 1:] = Theta1_grad[:, 1:] + lamb * Theta1[:, 1:] / m

    # Unroll gradients
    grad = np.append(Theta1_grad, Theta2_grad)

    return J, grad


def randInitializeWeights(L_in, L_out):
    epsilon_init = 0.12
    W = np.random.rand(L_out, 1 + L_in) * 2 * epsilon_init - epsilon_init
    return W


def debugInitializeWeights(fan_out, fan_in):
    W = np.zeros((fan_out, 1 + fan_in))
    W = np.reshape(np.sin(np.arange(1, (W.size + 1))), W.shape, order='F') / 10
    return W


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


def checkNNGradients(lamb=0):
    input_layer_size = 3
    hidden_layer_size = 5
    num_labels = 3
    m = 5
    Theta1 = debugInitializeWeights(hidden_layer_size, input_layer_size)
    Theta2 = debugInitializeWeights(num_labels, hidden_layer_size)
    X = debugInitializeWeights(m, input_layer_size - 1)
    y = 1 + np.mod(np.arange(1, m + 1), num_labels)[:, None]
    # Unroll parameters
    nn_params = np.append(Theta1[:], Theta2[:])
    costFunc = lambda p: nnCostFunction(p, input_layer_size, hidden_layer_size, num_labels, X, y, lamb)
    _, grad = costFunc(nn_params)
    numgrad = computeNumericalGradient(costFunc, nn_params)
    print(
        'The above two columns you get should be very similar.\n (Left-Your Numerical Gradient, Right-Analytical '
        'Gradient)\n\n')
    diff = np.linalg.norm(numgrad - grad) / np.linalg.norm(numgrad + grad)
    print('If your backpropagation implementation is correct, then \n'
          'the relative difference will be small (less than 1e-9). \n'
          '\nRelative Difference: {} \n'.format(diff))
    return


def training(X, y, input_layer_size, hidden_layer_size, num_labels, lamb):
    size = hidden_layer_size * (input_layer_size + 1) + num_labels * (hidden_layer_size + 1)
    all_theta = np.random.uniform(-0.12, 0.12, size)

    result = opt.minimize(fun=nnCostFunction, x0=all_theta,
                          args=(input_layer_size, hidden_layer_size, num_labels, X, y, lamb), method='TNC',
                          jac=True, options={'maxiter': 250})
    all_theta = result.x
    print(result)
    return all_theta


def predict(Theta1, Theta2, X):
    m = X.shape[0]
    X = np.hstack((np.ones((m, 1)), X))
    h1 = sigmoid(X.dot(Theta1.T))
    h1 = np.hstack((np.ones((m, 1)), h1))
    h2 = sigmoid(h1.dot(Theta2.T))

    p = np.argmax(h2, axis=1) + 1
    return p
