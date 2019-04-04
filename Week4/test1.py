from scipy import io
import numpy as np

from Week4 import Week4_assignment_in_python as week4


mat_file = io.loadmat('../resource/ex3data1.mat')

X = mat_file['X']
y = mat_file['y']

# data
data = np.array(X)
# label
label = np.array(y)

test_theta = np.ones((data.shape[1], 1))

week4.derivative_cost_function_vec(test_theta, X, label)