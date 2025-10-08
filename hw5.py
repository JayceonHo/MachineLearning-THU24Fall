import numpy as np
from sklearn.tree import DecisionTreeClassifier
import  pandas as pd

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

num_fold = 5
num_sample = 5000
num_sample_per_fold = num_sample // num_fold
train_loss_list, valid_loss_list = [], []
all_index = list(range(num_sample))


criterion = "entropy" # gini or entropy
splitter = "best" # best or random
max_depth = 1 # the maximal depth of the tree

train_acc_list, valid_acc_list = [], []
for i in range(num_fold):
    selected_index = all_index[i * num_sample_per_fold: (i + 1) * num_sample_per_fold]
    data_index = all_index[:i * num_sample_per_fold] + all_index[(i + 1) * num_sample_per_fold:]
    train_data, valid_data = train_data1[data_index], train_data1[selected_index]
    train_label, valid_label = train_label1[data_index], train_label1[selected_index]
    clf = DecisionTreeClassifier(criterion=criterion, splitter=splitter, max_depth=max_depth)
    clf.fit(train_data, train_label)
    train_loss_list.append(clf.score(train_data, train_label))
    valid_loss_list.append(clf.score(valid_data, valid_label))



# 评估模型
clf = DecisionTreeClassifier(criterion=criterion, splitter=splitter, max_depth=max_depth)
clf.fit(train_data1, train_label1)
accuracy = clf.score(test_data1, test_label1)
print(transform_data(train_loss_list))
print(transform_data(valid_loss_list))
print(f"Model accuracy: {accuracy:.2f}")

# # 可视化决策树
# plt.figure(figsize=(12,12))
# tree.plot_tree(clf, filled=True, feature_names=None, class_names=None)
# plt.show()
