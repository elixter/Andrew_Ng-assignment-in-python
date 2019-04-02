import numpy as np


# sigmoid function
def sigmoid(z):
    g = 1 / (1 + np.exp(np.negative(z)))

    return g


def hypothesis(theta, x):
    return sigmoid(np.sum(np.transpose(theta) * x))


# costFunction for Logistic Regression
def costFunction(theta, x, y):
    m = x.shape[0]
    sum = 0
    for i in range(m):
        sum += -y[i] * np.log(hypothesis(theta, x[i])) - (1 - y[i]) * np.log(1 - hypothesis(theta, x[i]))

    return sum / m


# theta j 에 대한 편미분값
def deriveCost(theta, x, y, j):
    m = x.shape[0]
    n = x.shape[1]
    J = 0

    for i in range(m):
        J += (hypothesis(theta, x[i]) - y[i]) * x[i][j]

    return J / m


# alpha == learning rate
def Gradient_Descent(theta, x, y, alpha=0.001):
    n = theta.shape[0]
    tmp = np.zeros(theta.shape, dtype=np.float32)

    # theta 업데이트는 한번에 해줘야하기때문에 tmp에 임시 저장
    for i in range(n):
        tmp[i][0] = theta[i][0] - alpha * deriveCost(theta, x, y, i)

    theta = tmp

    return theta


def Logistic_Regression(theta, x, y, epoch=400, alpha=0.001):
    tmp = costFunction(theta, x, y)

    # epoch 는 학습 반복 횟수
    for i in range(epoch):
        tmp_theta = Gradient_Descent(theta, x, y, alpha)
        tmp2 = costFunction(tmp_theta, x, y)

    # costFunction의 값이 최소가 되어야 하므로 이전 값 보다 작을 때 업데이트 시켜줌
        if tmp > tmp2:
            theta = tmp_theta
            tmp = tmp2

    return theta





