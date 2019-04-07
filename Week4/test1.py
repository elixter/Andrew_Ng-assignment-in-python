from scipy import io
import numpy as np

from Week4 import Week4_assignment_in_python as week4
from Week3 import Week3_assignment_in_python as week3


mat_file = io.loadmat('../resource/ex3data1.mat')

X = mat_file['X']
y = mat_file['y']

# data
data = np.array(X, dtype=np.float32)
# label
label = np.array(y, dtype=np.float32)



#data = data / data.max()

test_theta = np.random.randn(data.shape[1], 1) + 1

print(week4.cost_function_vec(test_theta, data, label))
print(week4.derivative_cost_function_vec(test_theta, data, label))
