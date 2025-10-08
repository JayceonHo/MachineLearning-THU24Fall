"""
This code is implemented for HW2 at Tsinghua SIGS
Using MLP for binary classification
"""
import torch.nn.functional as F
import  pandas as pd
import  torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np


def normalize(data):
    data_mean, data_std = data.mean(0), data.std(0)
    new_data = (data - data_mean) / (data_std + 1e-5)
    return new_data


class MLP(nn.Module):
    def __init__(self, input_channel, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_channel, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        # self.lr = nn.LeakyReLU()
        self.lr = nn.ReLU()
        # self.lr = nn.Tanh()
        # self.lr = nn.GELU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.lr(self.fc1(x))
        x = self.lr(self.fc3(x))
        x = F.sigmoid(self.fc2(x))
        return x

class MLP2(nn.Module):
    def __init__(self, input_channel, hidden_size, output_size):
        super(MLP2, self).__init__()
        self.fc1 = nn.Parameter(torch.randn(input_channel, hidden_size))
        self.fc2 = nn.Parameter(torch.randn(hidden_size, output_size))
        self.bias1 = nn.Parameter(torch.randn(hidden_size))
        self.bias2 = nn.Parameter(torch.randn(output_size))
    def forward(self, x):
        x = F.relu(torch.matmul(x, self.fc1) + self.bias1)
        x = F.sigmoid(torch.matmul(x, self.fc2) + self.bias2)
        return x

data_path = "./data1/" # input your data path
train_data1 = pd.read_csv(data_path + 'train1_icu_data.csv')
train_label1 = pd.read_csv(data_path + 'train1_icu_label.csv')
test_data1 = pd.read_csv(data_path + 'test1_icu_data.csv')
test_label1 = pd.read_csv(data_path + 'test1_icu_label.csv')

train_data1, train_label1 = torch.from_numpy(normalize(train_data1.to_numpy())).cuda().float(), torch.from_numpy(train_label1.to_numpy()).cuda().float() # 5000x108
test_data1, test_label1 = torch.from_numpy(normalize(test_data1.to_numpy())).cuda().float(), test_label1.to_numpy() # 1097x108

epoch = 100
# implement binary cross entropy loss by ourselves
# loss_func = lambda y1, y2: (-1/y1.shape[0]) * torch.sum((y1*torch.log(y2+1e-4) + (1-y1) * torch.log(1 - y2 + 1e-4)), dim=1)
loss_func =  nn.BCELoss()
num_sample = 5000
num_fold = 5
num_sample_per_fold = num_sample // num_fold
train_loss_list, valid_loss_list = [], []
all_index = list(range(num_sample))

train_error, valid_error = [], []
def transform_data(x):
    o = ""
    for i in range(len(x)):
        y = str(round(x[i] * 100))
        o += y + "\% "
        o+= "& "
    o += str(round(np.mean(x) * 100)) + "\%"
    return o

cal_accuracy = lambda y1, y2: np.mean([1 if y1[i] == y2[i] else 0 for i in range(len(y1))])


for i in range(num_fold):
    model = MLP(108, 50, 1).to("cuda")
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[250, 280, 300], gamma=0.5)
    selected_index = all_index[i * num_sample_per_fold: (i + 1) * num_sample_per_fold]
    data_index = all_index[:i * num_sample_per_fold] + all_index[(i + 1) * num_sample_per_fold:]
    train_data, valid_data = train_data1[data_index], train_data1[selected_index]
    train_label, valid_label = train_label1[data_index], train_label1[selected_index]
    for j in range(epoch):
        model.train()
        optimizer.zero_grad()
        output = model(train_data)
        train_loss = loss_func(output, train_label)
        train_loss.backward()
        optimizer.step()
        train_loss_list.append(train_loss.item())
        model.eval()
        with torch.no_grad():
            output = model(valid_data)
        valid_loss = loss_func(output, valid_label)
        valid_loss_list.append(valid_loss.item())
        scheduler.step()
    with torch.no_grad():
        train_output = (model(train_data).detach().cpu().numpy() > 0.5).astype("int")
        valid_output = (model(valid_data).detach().cpu().numpy() > 0.5).astype("int")
    train_error.append(cal_accuracy(train_output, train_label.detach().cpu().numpy()))
    valid_error.append(cal_accuracy(valid_output, valid_label.detach().cpu().numpy()))


model = MLP(108, 50, 1).to("cuda")
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
model.train()
for j in range(epoch):
    optimizer.zero_grad()
    output = model(train_data1)
    train_loss = loss_func(output, train_label1)
    train_loss.backward()
    optimizer.step()

# assert 1==2
# the following code is written for visualization of learning curve
# model.eval()
# prediction = (model(test_data1).detach().cpu().numpy() > 0.5).astype("int")
# accuracy = cal_accuracy(prediction, test_label1)
# fig, axs = plt.subplots(1, 5, figsize=(18, 3))
# cnt = 0
# ## visualization
# while cnt < 5:
#     x_index = list(range(epoch))
#     y1, y2 = train_loss_list[cnt*epoch:(cnt+1)*epoch], valid_loss_list[cnt*epoch:(cnt+1)*epoch]
#     axs[cnt].plot(x_index, y1, label="training loss")
#     axs[cnt].plot(x_index, y2, linestyle='--', label='validation loss')
#     axs[cnt].set_title("Fold-"+str(cnt+1))
#     # plt.title("Loss curve of training and validation")
#     axs[cnt].set_xlabel("Epoch")
#     axs[cnt].set_ylabel("Loss")
#     axs[cnt].legend()
#     axs[cnt].grid()
#     cnt+=1
#
# plt.tight_layout()
# plt.show()

