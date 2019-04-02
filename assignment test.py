import Week3_assignment_in_python as week3
import csv
import numpy as np
import matplotlib.pyplot as plt

data = []

with open('resource/ex2data1.csv', 'r', encoding='UTF8') as file:
    csv_reader = csv.reader(file, delimiter=',')
    for row in csv_reader:
        data_info = [float(row[0]), float(row[1]), int(row[2])]
        data.append(data_info)

admit_x1 = []
admit_x2 = []
non_admit_x1 = []
non_admit_x2 = []

for row in data:
    if row[2] == 1:
        admit_x1.append(float(row[0]))
        admit_x2.append(float(row[1]))
    else:
        non_admit_x1.append(float(row[0]))
        non_admit_x2.append(float(row[1]))


# 데이터 시각화
# plt.figure()
#
# plt.xlabel('Exam 1 score')
# plt.ylabel('Exam 2 score')
# plt.scatter(admit_x1, admit_x2, marker='+')
# plt.scatter(non_admit_x1, non_admit_x2, marker='o')
# plt.show()


initial_theta = np.zeros([len(data[0]), 1], dtype=np.float32)
test_theta = np.array([[-24],
                       [0.2],
                       [0.2]])

x_data = []
y_data = []

for row in data:
    set = [1, row[0], row[1]]
    x_data.append(set)
    y_data.append(row[2])

result = week3.costFunction(initial_theta, np.array(x_data), np.array(y_data))
print(result)

x = range(100)


result_theta = test_theta

result_theta = week3.Logistic_Regression(result_theta, np.array(x_data), np.array(y_data), alpha=0.001)

# 시각화
plt.figure()
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')
plt.scatter(admit_x1, admit_x2, marker='+')
plt.scatter(non_admit_x1, non_admit_x2, marker='o')
plt.axis([0, 100, 0, 100])
plt.plot(x, -((result_theta[0][0] + result_theta[1][0] * x) / result_theta[2][0]))
plt.show()
