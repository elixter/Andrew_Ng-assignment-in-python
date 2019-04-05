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


# 이부분 부터 다시 생각하며 고쳐야함
def derivative_cost_function_vec(theta, x, y):
    m, n = x.shape

    d = np.zeros((n, 1))
    h = np.zeros((n, 1))

    for i in range(m):
        h[i] = hypothesis(theta, x[i]) - y[i]

    d = (1 / m) * np.dot(np.transpose(x), h)

    return d


# rate means lambda
def derivative_cost_function_reg_vec(theta, x, y, rate=0.01):
    m, n = x.shape

    dJ = derivative_cost_function_vec(theta, x, y)

    for j in range(1, n):
        dJ[j] += rate / m * theta[j]

    return dJ


def gradient_descent_reg_vec(theta, x, y, rate=0.01, alpha=0.001):
    m, n = x.shape

    theta = np.dot(alpha, derivative_cost_function_reg_vec(theta, x, y, rate))

    return theta


def logistic_regression_reg_vec(theta, x, y, rate=0.01, alpha=0.001, epoch=400)
    m, n = x.shape

    tmp_cost = cost_function_vec(x, theta)

    for i in range(epoch):
        tmp_theta = gradient_descent_reg_vec(theta, x, y, rate, alpha)

        tmp_cost =



def one_vs_all_classification(theta, x, y):
