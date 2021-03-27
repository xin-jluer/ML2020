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
x_new = x.copy()
# Normalize
mean_x = np.mean(x, axis=0)  # 18 * 9
std_x = np.std(x, axis=0)  # 18 * 9
for i in range(len(x)):  # 12 * 471
    for j in range(len(x[0])):  # 18 * 9
        if std_x[j] != 0:
            x[i][j] = (x[i][j] - mean_x[j]) / std_x[j]

# 求mean和std的另一种思路，由于18*9个feature中，每9个feature都是属于同一种大气成分，因此可以9个一组进行求mean和std，而不是1个feature求一个mean和std
# 因此总共有18种物质，即18种mean和std
# 先初始化为0
mean_x_new = np.zeros(18)
std_x_new = np.zeros(18)
# 每9列分为一组求mean和std,共18组
for i in range(18):
    mean_x_new[i] = np.mean(x_new[:, 9*i : 9*(i+1)])
    std_x_new[i] = np.std(x_new[:, 9*i : 9*(i+1)])

# 对feature进行归一化处理
for i in range(18):
    if std_x_new[i] != 0:
        x_new[:, 9*i : 9*(i+1)] = (x_new[:, 9*i : 9*(i+1)] - mean_x_new[i]) / std_x_new[i]

# 查看使用新的normalize后的x_new
x = x_new

x_train_set = x[: math.floor(len(x) * 0.8), :]
y_train_set = y[: math.floor(len(y) * 0.8), :]
x_validation = x[math.floor(len(x) * 0.8):, :]
y_validation = y[math.floor(len(y) * 0.8):, :]


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

for learning_rate in [4]:
    w = np.zeros([dim, 1])
    eps = 0.0000000001
    iter_time = 100000
    adagrad = np.zeros([dim, 1])
    last_loss_dev = 0  # rmse
    loss_array = []
    iter_array = []
    for t in range(iter_time):
        loss = np.sqrt(np.sum(np.power(np.dot(x_train_set, w) - y_train_set, 2)) / 4521)  # rmse
        loss_dev = np.sqrt(np.sum(np.power(np.dot(x_validation, w) - y_validation, 2)) / 1131)  # rmse
        if t % 100 == 0:
            print("loss on train " + str(t) + ": " + str(loss))
            loss_array.append(loss)
            iter_array.append(t)
            if loss_dev >= last_loss_dev and last_loss_dev != 0:
                print("loss on valid increase, break " + str(t) + ": " + str(loss_dev))
                break
            else:
                last_loss_dev = loss_dev
                print("loss on valid " + str(t) + ": " + str(loss_dev))
        gradient = 2 * np.dot(x_train_set.transpose(), np.dot(x_train_set, w) - y_train_set)  # dim*1
        adagrad += gradient ** 2
        w = w - learning_rate * gradient / np.sqrt(adagrad + eps)
    np.save('weight', w)
    plt.plot(iter_array, loss_array, label=str(learning_rate))
plt.legend()
plt.show()


# preprocess test
testdata = pd.read_csv('./test.csv', header=None, encoding='big5')
test_data = testdata.iloc[:, 2:]
test_data[test_data == 'NR'] = 0
test_data = test_data.to_numpy()
test_x = np.empty([240, 18 * 9], dtype=float)
for i in range(240):
    test_x[i, :] = test_data[18 * i: 18 * (i + 1), :].reshape(1, -1)
for i in range(len(test_x)):
    for j in range(len(test_x[0])):
        if std_x[j] != 0:
            test_x[i][j] = (test_x[i][j] - mean_x[j]) / std_x[j]
test_x = np.concatenate((np.ones([240, 1]), test_x), axis=1).astype(float)

# Prediction
w = np.load('weight.npy')
ans_y = np.dot(test_x, w)

# Save to CSV File
with open('submit.csv', mode='w', newline='') as submit_file:
    csv_writer = csv.writer(submit_file)
    header = ['id', 'value']
    print(header)
    csv_writer.writerow(header)
    for i in range(240):
        row = ['id_' + str(i), ans_y[i][0]]
        csv_writer.writerow(row)
        print(row)
