import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import csv

# load train.csv
data = pd.read_csv('./train.csv', encoding='big5')
# Preprocessing
data = data.iloc[:, 3:]
data[data == 'NR'] = 0
raw_data = data.to_numpy()

# Extract Features
month_data = {}
for month in range(12):
    sample = np.empty([18, 480])
    for day in range(20):
        sample[:, day * 24: (day + 1) * 24] = raw_data[18 * (20 * month + day): 18 * (20 * month + day + 1), :]
    month_data[month] = sample

x = np.empty([12 * 471, 18 * 9], dtype=float)
y = np.empty([12 * 471, 1], dtype=float)
for month in range(12):
    for day in range(20):
        for hour in range(24):
            if day == 19 and hour > 14:
                continue
            x[month * 471 + day * 24 + hour, :] = \
                month_data[month][:, day * 24 + hour: day * 24 + hour + 9].reshape(1, -1)
            # vector dim:18*9 (9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9)
            y[month * 471 + day * 24 + hour, 0] = month_data[month][9, day * 24 + hour + 9]  # value
# print(x)
# print(y)

# Normalize
mean_x = np.mean(x, axis=0)  # 18 * 9
std_x = np.std(x, axis=0)  # 18 * 9
for i in range(len(x)):  # 12 * 471
    for j in range(len(x[0])):  # 18 * 9
        if std_x[j] != 0:
            x[i][j] = (x[i][j] - mean_x[j]) / std_x[j]

x_train_set = x[: math.floor(len(x) * 0.8), :]
y_train_set = y[: math.floor(len(y) * 0.8), :]
x_validation = x[math.floor(len(x) * 0.8):, :]
y_validation = y[math.floor(len(y) * 0.8):, :]

x_train_set5 = x_train_set[:, 18 * 4: 18 * 9]
x_validation5 = x_validation[:, 18 * 4: 18 * 9]
print(x_train_set5)
# print(x_train_set)
# print(y_train_set)
# print(x_validation)
# print(y_validation)
print(len(x_train_set))
print(len(y_train_set))  # 4521
print(len(x_validation))
print(len(y_validation))  # 1131

# Training
dim = 18 * 9 + 1
x_train_set = np.concatenate((np.ones([4521, 1]), x_train_set), axis=1).astype(float)
x_validation = np.concatenate((np.ones([1131, 1]), x_validation), axis=1).astype(float)

dim5 = 18 * 5 + 1
x_train_set5 = np.concatenate((np.ones([4521, 1]), x_train_set5), axis=1).astype(float)
x_validation5 = np.concatenate((np.ones([1131, 1]), x_validation5), axis=1).astype(float)

learning_rate = 5
w = np.zeros([dim, 1])
iter_time = 10000
adagrad = np.zeros([dim, 1])
eps = 0.0000000001
losst = {}
losst_v = 0  # rmse
for t in range(iter_time):
    loss = np.sqrt(np.sum(np.power(np.dot(x_train_set, w) - y_train_set, 2)) / 4521)  # rmse
    loss_v = np.sqrt(np.sum(np.power(np.dot(x_validation, w) - y_validation, 2)) / 1131)  # rmse
    if t % 100 == 0:
        print("loss on train:" + str(t) + ":" + str(loss))
        losst[t] = loss
        if loss_v > losst_v and losst_v != 0:
            print("loss on valid increase, break" + str(t) + ":" + str(loss_v))
            break
        else:
            losst_v = loss_v
            print("loss on valid:" + str(t) + ":" + str(loss_v))
    gradient = 2 * np.dot(x_train_set.transpose(), np.dot(x_train_set, w) - y_train_set)  # dim*1
    adagrad += gradient ** 2
    w = w - learning_rate * gradient / np.sqrt(adagrad + eps)

    # 求出 在 valid 的 loss ，如果 loss 大于之前的 ， 就 使用 之前的 w
names = list(losst.keys())
values = list(losst.values())
plt.plot(names, values, label="18 * 9")

plt.legend()

learning_rate = 2
w = np.zeros([dim5, 1])
iter_time = 10000
adagrad = np.zeros([dim5, 1])
eps = 0.0000000001
losst_5 = {}
losst_v = 0  # rmse
for t in range(iter_time):
    loss = np.sqrt(np.sum(np.power(np.dot(x_train_set5, w) - y_train_set, 2)) / 4521)  # rmse
    loss_v = np.sqrt(np.sum(np.power(np.dot(x_validation5, w) - y_validation, 2)) / 1131)  # rmse
    if t % 100 == 0:
        print("loss on train:" + str(t) + ":" + str(loss))
        losst_5[t] = loss
        if loss_v > losst_v and losst_v != 0:
            print("loss on valid increase, break" + str(t) + ":" + str(loss_v))
            break
        else:
            losst_v = loss_v
            print("loss on valid:" + str(t) + ":" + str(loss_v))
    gradient = 2 * np.dot(x_train_set5.transpose(), np.dot(x_train_set5, w) - y_train_set)  # dim*1
    adagrad += gradient ** 2
    w = w - learning_rate * gradient / np.sqrt(adagrad + eps)

    # 求出 在 valid 的 loss ，如果 loss 大于之前的 ， 就 使用 之前的 w

names2 = list(losst_5.keys())
values2 = list(losst_5.values())
plt.plot(names2, values2, label="18 * 5")
plt.legend()
plt.show()

