import Week3_assignment_in_python as week3
import csv
import numpy as np
import matplotlib.pyplot as plt

data = []

with open('resource/ex2data1.csv', 'r', encoding='UTF8') as file:
    csv_reader = csv.reader(file, delimiter=',')
    for row in csv_reader:
        data_info = (row[0], row[1], row[2])
        data.append(data_info)

admit_x = []
admit_y = []
non_admit_x = []
non_admit_y = []

for row in data:
    if row[2] == '1':
        admit_x.append(float(row[0]))
        admit_y.append(float(row[1]))
    else:
        non_admit_x.append(float(row[0]))
        non_admit_y.append(float(row[1]))


# 데이터 시각화
plt.figure()

plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')
plt.scatter(admit_x, admit_y, marker='+')
plt.scatter(non_admit_x, non_admit_y, marker='o')
plt.show()


print(np.multiply([1, 2, 3, 4,], [2, 2, 2, 2]))