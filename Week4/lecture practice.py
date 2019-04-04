import numpy as np

theta1 = np.ones([10, 11])
theta2 = 2 * np.ones([10, 11])
theta3 = 3 * np.ones([1, 11])

theta_vec = np.hstack((theta1.flatten(), theta2.flatten(), theta3.flatten()))

n_theta1 = np.reshape(theta_vec[0:110], (10, 11))
print(n_theta1 == theta1)