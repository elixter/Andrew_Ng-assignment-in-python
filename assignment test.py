import Week3_assignment_in_python as week3
import csv
import numpy as np

data = []

with open('resource/ex2data1.csv', 'r', encoding='UTF8') as file:
    csv_reader = csv.reader(file, delimiter=',')
    for row in csv_reader:
        data_info = (row[0], row[1])
        data.append(data_info)
        print(row[0], row[1])


print(data)

_data = np.array(data)
initial_theta = np.zeros(_data.shape)
print(_data.shape)

xTrain = np.zeros(_data.shape[0])
yTrain = np.zeros(_data.shape[0])

print(xTrain.shape)
print(yTrain.shape)