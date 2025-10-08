from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import  pandas as pd
import numpy as np


def normalize(data):
    data_mean, data_std = data.mean(0), data.std(0)
    new_data = (data - data_mean) / (data_std + 1e-5)
    return new_data

def transform_data(x):
    if x is []:
        return x
    o = ""
    for i in range(len(x)):
        y = str(round(x[i] * 100))
        o += y + "\% "
        o+= "& "
    o += str(round(np.mean(x) * 100)) + "\%"
    return o

def transform_mini_risk(y_prob):
    lr = np.log2(5 * y_prob[:, 0] / y_prob[:, 1])
    y_pred_risk = [0 if l > 0 else 1 for l in lr]
    return y_pred_risk

data_path = "./data1/" # input your data path
train_data1 = pd.read_csv(data_path + 'train1_icu_data.csv')
train_label1 = pd.read_csv(data_path + 'train1_icu_label.csv')
test_data1 = pd.read_csv(data_path + 'test1_icu_data.csv')
test_label1 = pd.read_csv(data_path + 'test1_icu_label.csv')

train_data1, train_label1 = normalize(train_data1.to_numpy()), train_label1.to_numpy()# 5000x108
test_data1, test_label1 = normalize(test_data1.to_numpy()), test_label1.to_numpy() # 1097x108

num_fold = 5
num_sample = 5000
num_sample_per_fold = num_sample // num_fold
train_loss_list, valid_loss_list = [], []
all_index = list(range(num_sample))

prior_prob = None # [0.2,0.8]
train_acc_list, valid_acc_list = [], []
for i in range(num_fold):
    selected_index = all_index[i * num_sample_per_fold: (i + 1) * num_sample_per_fold]
    data_index = all_index[:i * num_sample_per_fold] + all_index[(i + 1) * num_sample_per_fold:]
    train_data, valid_data = train_data1[data_index], train_data1[selected_index]
    train_label, valid_label = train_label1[data_index], train_label1[selected_index]
    gnb = GaussianNB(priors=prior_prob)
    gnb.fit(train_data, train_label)
    y_pred = gnb.predict(train_data)
    # y_prob = gnb.predict_proba(train_data)
    # y_pred_risk = transform_mini_risk(y_prob)
    # y_pred = y_pred_risk
    train_loss_list.append(accuracy_score(train_label, y_pred))
    y_pred = gnb.predict(valid_data)
    # y_prob = gnb.predict_proba(valid_data)
    # y_pred_risk = transform_mini_risk(y_prob)
    # y_pred = y_pred_risk
    valid_loss_list.append(accuracy_score(valid_label, y_pred))

gnb = GaussianNB(priors=prior_prob)
gnb.fit(train_data1, train_label1)
y_pred = gnb.predict(test_data1)
y_prob = gnb.predict_proba(test_data1)
y_pred_risk = transform_mini_risk(y_prob)

accuracy = accuracy_score(y_pred, test_label1)
accuracy_risk = accuracy_score(y_pred_risk, test_label1)
print(transform_data(train_loss_list))
print(transform_data(valid_loss_list))
print(accuracy, accuracy_risk)




