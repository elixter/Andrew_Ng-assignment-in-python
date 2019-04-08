import numpy as np


EPSILON = 0.0000001


def sigmoid(z):
    g = 1.0 / (1.0 + np.exp(np.negative(z)))
    if (g == 1.0):
        g -= EPSILON
    elif (g == 0.0):
        g += EPSILON

    return g


def hypothesis(theta, x):
    return sigmoid(np.dot(np.transpose(theta), np.transpose(x)))


# sigmoid function
def sigmoid_vec(z):
    g = 1.0 / (1.0 + np.exp(np.negative(z)))

    m = g.shape[0]

    for i in range(m):
        if (g[i] == 1):
            g[i] = g[i] - EPSILON
        elif (g[i] == 0):
            g[i] = g[i] + EPSILON

    return g


def hypothesis_vec(theta, x):
    h = sigmoid_vec(np.dot(np.transpose(theta), np.transpose(x)))

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

    h = sigmoid_vec(x_theta)
    J = 0

    J = np.sum((-y * np.log(h) - (1 - y) * np.log(1 - h)))

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

    h = sigmoid_vec(x_theta) - y

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
    tmp_theta = theta

    dj = derivative_cost_function_reg_vec(theta, x, y, rate)

    tmp_theta = theta - alpha * dj

    return tmp_theta


def logistic_regression_reg_vec(theta, x, y, rate=0.01, alpha=0.001, epoch=400):
    m, n = x.shape

    tmp_cost = cost_function_vec(theta, x, y)

    for i in range(epoch):
        tmp_theta = gradient_descent_reg_vec(theta, x, y, rate, alpha)
        cost = cost_function_vec(tmp_theta, x, y)
        if (tmp_cost > cost):
            theta = tmp_theta
            tmp_cost = cost

        process_percent = int(i / epoch * 100)
        proc_string = 'Learning Process(%) : ' + str(process_percent) + ' %'

        print(proc_string)

    print('Logistic Regression completed\n')

    return theta


def one_vs_all_classification(theta, x, y, K, rate=0.01, alpha=0.001, epoch=400):
    m, n = x.shape
    param = np.zeros((K + 1, n + 1))
    tmp_y = np.array(y)

    X = np.hstack((np.ones((m, 1)), x))

#     # 1 ~ K
    for k in range(1, K + 1):
        tmp_theta = np.vstack((np.zeros((1, )), theta))
        for i in range(m):
            if (tmp_y[i] != k):
                tmp_y[i] = 0
            elif (tmp_y[i] == k):
                tmp_y[i] = 1

        tmp_theta = logistic_regression_reg_vec(tmp_theta, X, tmp_y, rate, alpha, epoch)
        param[k] = tmp_theta.reshape((n + 1, ))

    return param


def one_vs_all_predict(param, x):
    n = param.shape[0]
    X = np.hstack((np.ones((1, )), x))

    prob = np.zeros((n, ))

    for i in range(0, n):
        #tmp_param = param[i].reshape((param[1].shape[0], 1))
        prob[i] = hypothesis(param[i], X)

    return prob
