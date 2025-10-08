import torch
import torch.nn as nn
import numpy as np
import random
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from hw7 import decode_idx3_ubyte, decode_idx1_ubyte
from hw2 import normalize

# normalize the dataset
def transform_data(x):
    o = ""
    for i in range(len(x)):
        y = str(round(x[i] * 100))
        o += y + "\% "
        o+= "& "
    o += str(round(np.mean(x) * 100)) + "\%"
    return o

# build a CNN model
class CNN(nn.Module):
    def __init__(self, hidden_dim=32, kernel_size=3, padding=1, stride=1):
        super(CNN, self).__init__()
        self.conv_net = nn.Sequential(
            nn.Conv2d(1, hidden_dim, kernel_size=kernel_size, stride=stride, padding=padding),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            nn.AvgPool2d(kernel_size=2,stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dim, 1, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.LeakyReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(49, 10),
        )
    def forward(self, x):
        x = self.conv_net(x)
        # assert 1==2, print(x.shape)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

# load the data and load from given files
raw_data, label = (decode_idx3_ubyte('data2_raw/train-images-idx3-ubyte'),
                   decode_idx1_ubyte('data2_raw/train-labels-idx1-ubyte'))
data = normalize(raw_data)
data = torch.from_numpy(data).float().to("cuda").unsqueeze(1)
label = torch.from_numpy(label).long().to("cuda")
train_index = random.sample(list(range(data.shape[0])), int(data.shape[0] * 0.9))
valid_index = random.sample(train_index, int((data.shape[0] * 0.1)))
test_index = list(set(list(range(data.shape[0]))) - set(train_index))
train_index = list(set(train_index) - set(valid_index))
train_data, valid_data, test_data = data[train_index], data[valid_index], data[test_index]
train_label, valid_label, test_label = label[train_index], label[valid_index], label[test_index]

# hyper-parameters setting
hidden_dim = 32
kernel_size = 7
stride = 1
padding = 3
num_repeat = 1
epoch = 20
batch_size = 128
loss_func = nn.CrossEntropyLoss()
train_acc, valid_acc = [], []

# training loop
for _ in range(num_repeat):
    model = CNN(hidden_dim, kernel_size, padding, stride).to("cuda")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_loss, valid_loss = [], []
    for i in range(epoch):
        loss_list = []
        model.eval()
        for j in range(0, valid_data.shape[0], batch_size):
            img = valid_data[j:j + batch_size, ...]
            in_label = valid_label[j:j + batch_size]
            with torch.no_grad():
                output = model(img)
            loss = loss_func(output, in_label)
            loss_list.append(loss.item())
        valid_loss.append(np.mean(loss_list))
        loss_list = []
        model.train()
        for j in range(0, train_data.shape[0], batch_size):
            img = train_data[j:j+batch_size,...]
            in_label = train_label[j:j+batch_size]
            optimizer.zero_grad()
            output = model(img)

            loss = loss_func(output, in_label)
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
        train_loss.append(np.mean(loss_list))

    model.eval()
    with torch.no_grad():
        pred = model(train_data)
        pred = pred.argmax(dim=1)
        train_acc.append(pred.eq(train_label).float().mean().item())

        pred = model(valid_data)
        pred = pred.argmax(dim=1)
        valid_acc.append(pred.eq(valid_label).float().mean().item())

print("train accuracy: ", transform_data(train_acc))
print("valid accuracy:", transform_data(valid_acc))


# do model evaluation
with torch.no_grad():
    pred = model(test_data)
    pred = pred.argmax(dim=1)
    correct = round(pred.eq(test_label).float().mean().item() * 100.)
    pred = pred.cpu().detach().numpy()
    test_label = test_label.cpu().detach().numpy()
    cm = confusion_matrix(test_label, pred)
    precision, recall, f1, _ = precision_recall_fscore_support(test_label, pred, average=None)

# print the results
print("accuracy is: ", correct)
print("F1 score is: ", f1)
print("Precision is: ", precision)
print("Recall is: ", recall)
print("confusion matrix is: ", cm)

# plot the learning curve
plt.plot(list(range(len(train_loss))), train_loss, 'b', label='train loss')
plt.plot(list(range(len(valid_loss))), valid_loss, 'r', label='validation loss')
plt.legend()
plt.grid(linestyle="-.")
plt.xticks(list(range(1,len(train_loss)+1)))
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title("Learning Curve of CNN")
plt.show()


# calculate other metrics
f1, precision, recall = map(lambda x:[round(i*100) for i in  x], (precision, recall, f1))
plt.plot(list(range(len(f1))), f1, marker="o")
plt.grid(axis="y")
plt.xticks(list(range(1,len(f1)+1)))
plt.title("F1 Score of CNN on Test Set")
plt.xlabel("Class")
plt.ylabel("F1 Score")
plt.show()

# report and visualize the results
plt.plot(list(range(len(recall))), recall, marker="o")
plt.grid(axis="y")
plt.xticks(list(range(1,len(recall)+1)))
plt.title("Recall of CNN on Test Set")
plt.xlabel("Class")
plt.ylabel("Recall")
plt.show()

plt.plot(list(range(len(precision))), precision, marker="o")
plt.grid(axis="y")
plt.xticks(list(range(1,len(recall)+1)))
plt.title("Precision of CNN on Test Set")
plt.xlabel("Class")
plt.ylabel("Recall")
plt.show()

for y in range(cm.shape[0]):
    for x in range(cm.shape[1]):
        plt.text(y, x, cm[y][x], horizontalalignment='center', verticalalignment='center')
plt.imshow(cm, cmap='GnBu')
plt.colorbar()
plt.xlabel('Prediction')
plt.ylabel('Label')
plt.xticks(range(1,11))
plt.yticks(range(1,11))
plt.grid(False)
plt.title('Confusion Matrix of CNN on Test Set')
plt.show()

