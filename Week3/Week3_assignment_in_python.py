import numpy as np
import math


# sigmoid function
def sigmoid(z):
    g = 1.0 / (1.0 + np.exp(np.negative(z)))

    return g


def hypothesis(theta, x):
    return sigmoid(np.sum(np.transpose(theta) * x))


# costFunction for Logistic Regression
def cost_function(theta, x, y):
    m = x.shape[0]
    sum = 0
    for i in range(m):
        h = hypothesis(theta, x[i])
        sum += -y[i] * np.log(h) - (1 - y[i]) * np.log(1 - h)

    result = sum / m

    return result


# theta j 에 대한 편미분값
def derive_cost(theta, x, y, j):
    m = x.shape[0]
    n = x.shape[1]
    J = 0

    for i in range(m):
        J += (hypothesis(theta, x[i]) - y[i]) * x[i][j]

    result = J / m

    return result


# alpha == learning rate
def gradient_descent(theta, x, y, alpha=0.001):
    n = theta.shape[0]
    tmp = np.zeros(theta.shape, dtype=np.float32)

    # theta 업데이트는 한번에 해줘야하기때문에 tmp에 임시 저장
    for i in range(n):
        tmp[i][0] = theta[i][0] - alpha * derive_cost(theta, x, y, i)

    theta = tmp

    return theta


def logistic_regression(theta, x, y, epoch=400, alpha=0.001):
    tmp = cost_function(theta, x, y)

    # epoch 는 학습 반복 횟수
    for i in range(epoch):
        tmp_theta = gradient_descent(theta, x, y, alpha)
        tmp2 = cost_function(tmp_theta, x, y)

    # cost function의 값이 최소가 되어야 하므로 이전 값 보다 작을 때 업데이트 시켜줌
        if tmp > tmp2:
            theta = tmp_theta
            tmp = tmp2

        process_percent = int(i / epoch * 100)
        correct = 0

        for i in range(x.shape[0]):
            if (hypothesis(theta, x[i]) < 0.5 and y[i] == 0):
                correct += 1
            elif (hypothesis(theta, x[i]) >= 0.5 and y[i] == 1):
                correct += 1

        accuracy = correct / x.shape[0] * 100

        proc_string = 'Learning Process(%) : ' + str(process_percent) + ' %'
        acc_str = 'Accuracy(%) : ' + str(accuracy) + '%\n'

        print(proc_string)
        print(acc_str)

    return theta


def predict(theta, data):

    data_to_predict = np.array([1, data[0], data[1]], dtype=np.float32)

    result = hypothesis(theta, data_to_predict)
    print('Probability of admit is : ' + str(result))

    if (result < 0.5):
        print("Couldn't admit")
    else:
        print('Admit')


# ===================== Regularization =================================

def map_feature(data, degree=6):
    m = data.shape[0]
    n = data.shape[1]
    map = []

    for i in range(m):
        tmp = [1]
        for j in range(1, degree + 1):
            tmpval = 0
            for k in range(j + 1):
                tmpval = math.pow(data[i][0], j - k) * math.pow(data[i][1], k)

                tmp.append(tmpval)

        map.append(tmp)

    featured_map = np.array(map)

    return featured_map


def cost_function_reg(theta, x, y, rate=0.1):
    n = theta.shape[0]      # num of theta
    m = x.shape[0]          # num of data

    theta_sum = 0
    origin_cost = cost_function(theta, x, y)

    for j in range(n):
        theta_sum = theta[j][0] ** 2        # sum of theta(j) square

    result = origin_cost + rate / (2 * m) * theta_sum

    return result


def derive_cost_reg(theta, x, y, j, rate=0.1):
    m = x.shape[0]

    origin = derive_cost(theta, x, y, j)
    thetaj = rate / m * theta[j][0]

    result = origin + thetaj

    return result


def gradient_descent_reg(theta, x, y, alpha=0.001, rate=0.1):
    n = theta.shape[0]
    tmp = np.zeros(theta.shape, dtype=np.float32)

    # theta 업데이트는 한번에 해줘야하기때문에 tmp에 임시 저장
    for i in range(n):
        tmp[i][0] = theta[i][0] - alpha * derive_cost_reg(theta, x, y, i, rate)

    theta = tmp

    return theta


def logistic_regression_reg(theta, x, y, epoch=400, alpha=0.001, rate=0.1):
    tmp = cost_function_reg(theta, x, y, rate)

    # epoch 는 학습 반복 횟수
    for i in range(epoch):
        tmp_theta = gradient_descent_reg(theta, x, y, alpha, rate)
        tmp2 = cost_function_reg(tmp_theta, x, y)

        # cost function의 값이 최소가 되어야 하므로 이전 값 보다 작을 때 업데이트 시켜줌
        if tmp > tmp2:
            theta = tmp_theta
            tmp = tmp2

        process_percent = int(i / epoch * 100)

        correct = 0

        for i in range(x.shape[0]):
            if (hypothesis(theta, x[i]) < 0.5 and y[i] == 0):
                correct += 1
            elif (hypothesis(theta, x[i]) >= 0.5 and y[i] == 1):
                correct += 1

        accuracy = correct / x.shape[0] * 100

        proc_string = 'Learning Process(%) : ' + str(process_percent) + ' %\n'
        acc_str = 'Accuracy(%) : ' + str(accuracy) + '%\n'

        print(proc_string)
        print(acc_str)

    return theta


def predict_reg(theta, data):

    result = hypothesis(theta, data)
    print('Probability of admit is : ' + str(result))

    if (result < 0.5):
        print("Deny")
    else:
        print('Accept')
