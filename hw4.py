import torch.nn.functional as F
import  pandas as pd
import  torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

from sklearn import svm
from sklearn.metrics import classification_report, accuracy_score

def normalize(data):
    data_mean, data_std = data.mean(0), data.std(0)
    new_data = (data - data_mean) / (data_std + 1e-5)
    return new_data

def transform_data(x):
    o = ""
    for i in range(len(x)):
        y = str(round(x[i] * 100))
        o += y + "\% "
        o+= "& "
    o += str(round(np.mean(x) * 100)) + "\%"
    return o

data_path = "./data1/" # input your data path
train_data1 = pd.read_csv(data_path + 'train1_icu_data.csv')
train_label1 = pd.read_csv(data_path + 'train1_icu_label.csv')
test_data1 = pd.read_csv(data_path + 'test1_icu_data.csv')
test_label1 = pd.read_csv(data_path + 'test1_icu_label.csv')

train_data1, train_label1 = normalize(train_data1.to_numpy()), train_label1.to_numpy()# 5000x108
test_data1, test_label1 = normalize(test_data1.to_numpy()), test_label1.to_numpy() # 1097x108


num_sample = 5000
num_fold = 5
num_sample_per_fold = num_sample // num_fold
train_loss_list, valid_loss_list = [], []
all_index = list(range(num_sample))


degree = 7 # this controls the order of the polynomial, it works only when selected kernel is polynomial
kernel_list = ["rbf", "linear", "poly", "sigmoid"] # can
C = 1
kernel = kernel_list[2]
train_acc_list, valid_acc_list = [], []
for i in range(num_fold):
    selected_index = all_index[i * num_sample_per_fold: (i + 1) * num_sample_per_fold]
    data_index = all_index[:i * num_sample_per_fold] + all_index[(i + 1) * num_sample_per_fold:]
    train_data, valid_data = train_data1[data_index], train_data1[selected_index]
    train_label, valid_label = train_label1[data_index], train_label1[selected_index]
    clf = svm.SVC(kernel=kernel, C=C, gamma='scale', degree=degree)
    clf.fit(train_data, train_label)
    train_acc = accuracy_score(train_label, clf.predict(train_data))
    valid_acc = accuracy_score(valid_label, clf.predict(valid_data))
    train_loss_list.append(train_acc)
    valid_loss_list.append(valid_acc)


clf = svm.SVC(kernel='rbf', C=1.0, gamma='scale')
clf.fit(train_data1, train_label1)
y_pred = clf.predict(test_data1)

# 评估模型
# print(classification_report(test_label1, y_pred))
print(transform_data(train_loss_list))
print(transform_data(valid_loss_list))
print(f'Accuracy: {accuracy_score(test_label1, y_pred)}')
