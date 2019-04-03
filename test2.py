import Week3_assignment_in_python as week3
import csv
import numpy as np
import matplotlib.pyplot as plt

data = []

with open('resource/ex2data2.csv', 'r', encoding='UTF8') as file:
    csv_reader = csv.reader(file, delimiter=',')
    for row in csv_reader:
        data_info = [float(row[0]), float(row[1]), int(row[2])]
        data.append(data_info)

# 데이터 받아오기
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
plt.figure()

plt.xlabel('Microchip Test 1')
plt.ylabel('Microchip test 2')
plt.scatter(admit_x1, admit_x2, marker='+')
plt.scatter(non_admit_x1, non_admit_x2, marker='o')
plt.show()

initial_theta = np.zeros([len(data[0]), 1], dtype=np.float32)

x_data = []
y_data = []

# 학습을 위한 데이터 가공
for row in data:
    set = [1, row[0], row[1]]
    x_data.append(set)
    y_data.append(row[2])

x_data = np.array(x_data)
y_data = np.array(y_data)


result = week3.cost_function_reg(initial_theta, x_data, y_data)
print(result)

# 데이터 맵핑
mapped_data = week3.map_feature(x_data)

# theta 초기값 설정 매우 커지면 sigmoid 계산에서 언더플로우 발생
test_theta = np.random.randn(mapped_data.shape[1], 1) + 1

print(mapped_data.shape[1])

result = week3.logistic_regression_reg(test_theta, mapped_data, y_data)

valid_data = np.array([[-0.25, 1.5]]) # 검증? 데이터
mapped_valid_data = week3.map_feature(valid_data) # 데이터를 맵핑해준 후 분류기에 넣어야함

week3.predict_reg(result, week3.map_feature(mapped_valid_data))