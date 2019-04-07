import numpy as np


EPSILON = 0.0000001


# sigmoid function
def sigmoid(z):
    g = 1.0 / (1.0 + np.exp(np.negative(z)))

    m, n = g.shape
    for i in range(m):
        if (g[i] == 1):
            g[i] = g[i] - EPSILON
        elif (g[i] == 0):
            g[i] = g[i] + EPSILON

    return g


def hypothesis(theta, x):
    h = sigmoid(np.dot(np.transpose(theta), np.transpose(x)))

    return h


def cost_function_vec(theta, x, y):
    m, n = x.shape
    t_theta = theta
    if (theta.shape[0] != n):
        # theta transpose
        t_theta = np.transpose(theta)

    x_theta = np.zeros((x.shape[0], 1))

    # theta (1, n) * X (n, )
    for i in range(m):
       x_theta[i] = np.dot(np.transpose(t_theta), x[i])

    h = sigmoid(x_theta)
    J = 0

    J = np.sum((-y * np.log(h) - (1 - y) * np.log(1 - h)))

    print(J)

    J = J / m

    return J


# Partial derivative cost function
def derivative_cost_function_vec(theta, x, y):
    m, n = x.shape

    t_theta = theta
    if (theta.shape[0] != n):
        # theta transpose
        t_theta = np.transpose(theta)

    d = np.zeros((n, 1))
    h = np.zeros((n, 1))

    # for i in range(m):
    #     h[i] = hypothesis(theta, x[i]) - y[i]

    x_theta = np.zeros((x.shape[0], 1))

    # theta (1, n) * X (n, )
    for i in range(m):
        x_theta[i] = np.dot(np.transpose(t_theta), x[i])

    h = sigmoid(x_theta) - y

    d = (1 / m) * np.dot(np.transpose(x), h)

    return d


# rate means lambda
def derivative_cost_function_reg_vec(theta, x, y, rate=0.01):
    m, n = x.shape

    dJ = derivative_cost_function_vec(theta, x, y)

    for j in range(1, n):
        dJ[j] += rate / m * theta[j]

    return dJ


# 여기부터 고쳐야됨
def gradient_descent_reg_vec(theta, x, y, rate=0.01, alpha=0.001):
    m, n = x.shape

    theta = np.dot(alpha, derivative_cost_function_reg_vec(theta, x, y, rate))

    return theta


def logistic_regression_reg_vec(theta, x, y, rate=0.01, alpha=0.001, epoch=400):
    m, n = x.shape

    tmp_cost = cost_function_vec(x, theta)

    for i in range(epoch):
        tmp_theta = gradient_descent_reg_vec(theta, x, y, rate, alpha)

        #tmp_cost =



#def one_vs_all_classification(theta, x, y):
