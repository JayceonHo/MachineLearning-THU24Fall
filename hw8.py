import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from umap import UMAP

from hw7 import visualizing, pca_vis, decode_idx3_ubyte, decode_idx1_ubyte
from hw2 import normalize




raw_data, label = decode_idx3_ubyte('data2_raw/train-images-idx3-ubyte'), decode_idx1_ubyte('data2_raw/train-labels-idx1-ubyte')
digit = 8
index = [i for i in range(len(label)) if label[i] == digit]
raw_data = raw_data[index, :]
data = normalize(raw_data).reshape(raw_data.shape[0], -1)
data_2, _ = pca_vis(data, 2)
data_42, _ = pca_vis(data, 42)
umap = UMAP(n_neighbors=15, min_dist=0.1)
data_umap = umap.fit_transform(data.reshape(data.shape[0], -1))




corr_list = []
total_clusters = 51

# this part of code is for visualizing some samples in the class of digit 8
num_cluster = 10
cluster = KMeans(n_clusters=num_cluster)
cluster.fit(input_data)
pred_label = cluster.labels_
selected_index = []
for i in range(num_cluster):
    selected_index.append(np.where(pred_label == i)[0][0])
visualizing(data_umap, num_cluster, pred_label, "UMAP-1", "UMAP-2", "UMAP Plot")

# This part of code is used for evaluating the unsupervised K-means algorithm for digits classification problem
input_data = data_umap # it can be data, data_2, data_42
for num_clusters in range(2, total_clusters):
    cluster = KMeans(n_clusters=num_clusters)
    cluster.fit(input_data)
    labels = cluster.labels_
    corr_label = {}
    for i in range(num_clusters):
        t = []
        for j in range(len(labels)):
            if labels[j] == i:
                t.append(label[j])
        corr_label[i] = np.argmax(np.bincount(t))

    true_label = np.array([corr_label[i] for i in labels])
    corr = np.sum(((true_label - label) == np.zeros_like(label)).astype(np.float32))
    print(corr, corr/len(label))
    corr_list.append(corr)

plt.plot(list(range(2, total_clusters)), corr_list, marker="o")
plt.xlabel("The specified cluster number")
plt.xticks(list(range(2, total_clusters, int((total_clusters - 2)/20))))
plt.ylabel("The correct prediction over 60000 samples")
plt.title("The change of correction versus cluster numbers")
plt.grid(True, linestyle='--')
plt.show()

