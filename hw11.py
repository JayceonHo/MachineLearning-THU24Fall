"""
This is code for hw11 that is AG_News text classification with RNN
Author: Jiacheng Hou
Note: There are some functions and class referred from AI and Github repository
"""
import numpy as np
from torch import nn
import torch
from tqdm import tqdm
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import AG_NEWS
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torchtext.data.functional import to_map_style_dataset
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import matplotlib.pyplot as plt

class RNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim=64, num_class=4):
        super(RNN, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim)
        self.rnn = nn.RNN(1, hidden_dim, num_layers=2, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_class)

    def forward(self, text, offsets):
        """

        :param text: input digitalized text with shape of Nx1, N is the size of vocabulary table
        :param offsets: a sequence of marker to mark each batch with shape of B, B is batch size
        :return: class prediction Bx4
        """

        embedded = self.embedding(text, offsets) # return B x emb_size embedding vector
        out, hid = self.rnn(embedded.unsqueeze(-1))
        pred = self.fc(out).mean(1)
        # assert 1==2, print(out.shape, hid.shape)
        return pred

def predict(text, text_pipeline):
    with torch.no_grad():
        text = torch.tensor(text_pipeline(text))
        output = model(text, torch.tensor([0]))
        return output.argmax(1).item() + 1

def evaluate(dataloader):
    model.eval()
    total_acc = 0
    cm, precision, recall, f1 = np.zeros((num_class, num_class)), np.zeros(num_class), np.zeros(num_class), np.zeros(num_class)
    with torch.no_grad():
        for idx, (label, text, offsets) in enumerate(dataloader):
            weight = label.size(0)/len(dataloader.dataset)
            predicted_label = model(text, offsets)
            loss = loss_func(predicted_label, label)
            # assert 1==2, print(predicted_label, (predicted_label.argmax(1)).equal(label))
            total_acc += ((predicted_label.argmax(1)).eq(label)).float().mean().item() * weight
            label, predicted_label = label.detach().cpu().numpy(), predicted_label.detach().cpu().numpy()
            predicted_label = np.argmax(predicted_label, axis=1)
            cm += confusion_matrix(label, predicted_label)
            i_precision, i_recall, i_f1, _ = precision_recall_fscore_support(label, predicted_label, average=None)
            precision += i_precision * weight
            recall += i_recall * weight
            f1 += i_f1 * weight
        #assert 1==2, print(cnt,len(dataloader), dataloader.__len__())
    return total_acc, (cm, precision, recall, f1), loss.item()


def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text)

def collate_batch(batch):
    label_list, text_list, offsets = [], [], [0]
    for (_label, _text) in batch:
         label_list.append(label_pipeline(_label))
         processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
         text_list.append(processed_text)
         offsets.append(processed_text.size(0))
    label_list = torch.tensor(label_list, dtype=torch.int64)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text_list = torch.cat(text_list)
    return label_list.to(device), text_list.to(device), offsets.to(device)

def train(dataloader):
    model.train()
    for idx, (label, text, offsets) in enumerate(dataloader):
        optimizer.zero_grad()
        predicted_label = model(text, offsets)
        loss = loss_func(predicted_label, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()


device = "cuda"
# define tokenizer
tokenizer = get_tokenizer('basic_english')
train_iter = AG_NEWS(root="./data3/", split='train')

# using full train set to build vocabulary table that is a mapper, mapping words to numbers
vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=[""])
vocab.set_default_index(vocab[""])
print("The size of vocabulary table is : ", len(vocab))
"""
print("The length of vocabulary table is: ", len(vocab))
print("The number corresponds to here you are is: ", vocab(["here", "you", "are"]))
"""

text_pipeline = lambda x: vocab(tokenizer(x))
label_pipeline = lambda x: int(x) - 1

num_class = 4
vocab_size = len(vocab)
emb_size = 64
num_epoch = 10 # epoch
batch_size = 512 # batch size for training
loss_func = torch.nn.CrossEntropyLoss()
model = RNN(vocab_size, emb_size, num_class).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 6, gamma=0.3)
total_accu = None
train_iter, test_iter = AG_NEWS(root="./data3/")
train_dataset = to_map_style_dataset(train_iter)
test_dataset = to_map_style_dataset(test_iter)
# set 80% of raw train set as train set and remaining 20% as validation set, following 4:1 division of requirement
num_train = int(len(train_dataset) * 0.8)
num_val = len(train_dataset) - num_train
num_test = len(test_dataset)
# assert 1==2, print(len(test_dataset))
split_train_, split_valid_ = \
    random_split(train_dataset, [num_train, len(train_dataset) - num_train])

train_dataloader = DataLoader(split_train_, batch_size=batch_size,
                              shuffle=True, collate_fn=collate_batch)
valid_dataloader = DataLoader(split_valid_, batch_size=batch_size,
                              shuffle=True, collate_fn=collate_batch)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=True, collate_fn=collate_batch)

train_loss_list, valid_loss_list = [], []
for epoch in tqdm(range(1, num_epoch + 1)):
    train(train_dataloader)
    acc_train, _, loss_train = evaluate(train_dataloader)
    acc_val, _, loss_val = evaluate(valid_dataloader)
    train_loss_list.append(loss_train)
    valid_loss_list.append(loss_val)
    scheduler.step()

    print("\n The ", epoch, " epoch with training accuracy ", round(acc_train * 100), round(acc_val * 100))

plt.plot(list(range(1, num_epoch+1)), train_loss_list, "r", label="train")
plt.plot(list(range(1, num_epoch+1)), valid_loss_list, "b", label="valid")
plt.title("Learning Curve of RNN for AG_News Text Classification")
plt.xticks(list(range(1, num_epoch+1)))
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.legend()
plt.show()


correct, (cm, precision, recall, f1), _ = evaluate(test_dataloader)
print("accuracy is: ", correct)
print("F1 score is: ", f1)
print("Precision is: ", precision)
print("Recall is: ", recall)
print("confusion matrix is: ", cm)

class_tick = [1, 2, 3, 4]
# calculate other metrics
# f1, precision, recall = map(lambda x:[round(i*100) for i in  x], (precision, recall, f1))
plt.plot(class_tick, f1, marker="o")
plt.grid(axis="y")
plt.xticks(class_tick)
plt.title("F1 Score of CNN on Test Set")
plt.xlabel("Class")
plt.ylabel("F1 Score")
plt.show()

# report and visualize the results
plt.plot(class_tick, recall, marker="o")
plt.grid(axis="y")
plt.xticks(class_tick)
plt.title("Recall of CNN on Test Set")
plt.xlabel("Class")
plt.ylabel("Recall")
plt.show()

plt.plot(class_tick, precision, marker="o")
plt.grid(axis="y")
plt.xticks(class_tick)
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
plt.xticks([])
plt.yticks([])
plt.grid(False)
plt.title('Confusion Matrix of CNN on Test Set')
plt.show()



ag_news_label = {1: "World",
                 2: "Sports",
                 3: "Business",
                 4: "Sci/Tec"}



user_test_seq = "Download Trump was elected as US president last month."

model = model.to("cpu")

print("This is a %s news" %ag_news_label[predict(user_test_seq, text_pipeline)])
