import numpy as np
import matplotlib.pyplot as plt


# sigmoid function
def sigmoid(z):
    g = np.zeros(z.shape)
    g = 1 / (1 + np.exp(np.negative(z)))

    return g


def hypothesis(theta, x):
    return sigmoid(np.transpose(theta) * x)


# costFunction for Logistic Regression
def costFunction(theta, data, label):
    m = data.shape[0]
    sum = 0
    for i in range(m):
        sum += label[i] * np.log(hypothesis(theta, data[i])) - (1 - label[i]) * np.log(1 - hypothesis(theta, data[i]))

    return sum / m


# theta j 에 대한 편미분값
def deriveCost(theta, data, label):
    m = data.shape[0]
    J = 0

    for i in range(m):
        J += (hypothesis(theta, data[i]) - label[i]) * data[i]

    return J / m


#def Gradient_Descent(function):





test = np.array([1, 2, 3, 4, 5, 6])
test2 = np.array([1, 2, 3, 4, 5, 6])


print(costFunction([2], np.array([3]), np.array([3])))
