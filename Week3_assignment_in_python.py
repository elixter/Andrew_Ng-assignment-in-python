import numpy as np
import math


# sigmoid function
def sigmoid(z):
    g = 1 / (1 + np.exp(np.negative(z)))

    return g


def sigmoid(z):
    g = 1 / (1 + math.exp(-z))


def hypothesis(theta, x):
    return sigmoid(np.multiply(theta, x))


# costFunction for Logistic Regression
def costFunction(theta, x, y):
    m = x.shape[0]
    sum = 0
    for i in range(m):
        sum += y[i] * np.log(hypothesis(theta, x[i])) - (1 - y[i]) * np.log(1 - hypothesis(theta, x[i]))

    return sum / m


# theta j 에 대한 편미분값
def deriveCost(theta, x, y):
    m = x.shape[0]
    J = 0

    for i in range(m):
        J += (hypothesis(theta, x[i]) - y[i]) * x[i]

    return J / m


#def Gradient_Descent(function):




