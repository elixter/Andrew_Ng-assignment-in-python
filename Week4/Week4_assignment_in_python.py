import numpy as np

def sigmoid(z):
    g = 1 / (1 + np.exp(z))

    return g


def hypothesis(theta, x):
    h = sigmoid(np.dot(np.transpose(theta), x))

    return h


def cost_function_vec(X, theta):
    m, n = X.shape

    if (theta.shape[0] != n):
        # theta transpose
        t_theta = np.transpose(theta)

    X_theta = np.zeros((X.shape[0], 1))

    # theta (1, n) * X (n, )
    for i in range(n):
       X_theta[i] = np.dot(t_theta, X[i])

    return X_theta


def derivative_cost_function_vec(theta, x, y):
    m, n = x.shape

    d = np.zeros((n, 1))
    h = np.zeros((n, 1))

    for i in range(m):
        h[i] = hypothesis(theta, x[i]) - y[i]

    d = (1 / m) * np.dot(np.transpose(x), h)

    return d
