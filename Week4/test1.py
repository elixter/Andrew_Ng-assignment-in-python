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

m, n = data.shape

X = np.hstack((np.ones((m, 1)), data))

test_theta = np.random.randn(data.shape[1], 1) + 30


result_theta = week4.one_vs_all_classification(test_theta, data, label, 10)
result = week4.one_vs_all_predict(result_theta, tran_x[1])
print(result.argmax(), result[result.argmax()])
