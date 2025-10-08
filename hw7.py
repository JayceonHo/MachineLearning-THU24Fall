### code for MNIST data dimensionality reduction
### 2024 ML at Tsinghua SIGS
import numpy as np
import struct
from matplotlib import pyplot as plt
from umap import UMAP

## this part of code is referred from internet for data unpacking
def decode_idx3_ubyte(idx3_ubyte_file):
    """
    解析idx3文件的通用函数
    :param idx3_ubyte_file: idx3文件路径
    :return: 数据集
    """
    # 读取二进制数据
    bin_data = open(idx3_ubyte_file, 'rb').read()

    # 解析文件头信息，依次为魔数、图片数量、每张图片高、每张图片宽
    offset = 0
    fmt_header = '>iiii'
    magic_number, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, offset)
    print('魔数:%d, 图片数量: %d张, 图片大小: %d*%d' % (magic_number, num_images, num_rows, num_cols))

    # 解析数据集
    image_size = num_rows * num_cols
    offset += struct.calcsize(fmt_header)
    fmt_image = '>' + str(image_size) + 'B'
    images = np.empty((num_images, num_rows, num_cols))
    for i in range(num_images):
        if (i + 1) % 10000 == 0:
            print('已解析 %d' % (i + 1) + '张')
        images[i] = np.array(struct.unpack_from(fmt_image, bin_data, offset)).reshape((num_rows, num_cols))
        offset += struct.calcsize(fmt_image)
    return images

## this part of code for data reading is referred from internet
def decode_idx1_ubyte(idx1_ubyte_file):
    """
    解析idx1文件的通用函数
    :param idx1_ubyte_file: idx1文件路径
    :return: 数据集
    """
    # 读取二进制数据
    bin_data = open(idx1_ubyte_file, 'rb').read()

    # 解析文件头信息，依次为魔数和标签数
    offset = 0
    fmt_header = '>ii'
    magic_number, num_images = struct.unpack_from(fmt_header, bin_data, offset)
    print('魔数:%d, 图片数量: %d张' % (magic_number, num_images))

    # 解析数据集
    offset += struct.calcsize(fmt_header)
    fmt_image = '>B'
    labels = np.empty(num_images)
    for i in range(num_images):
        if (i + 1) % 10000 == 0:
            print('已解析 %d' % (i + 1) + '张')
        labels[i] = struct.unpack_from(fmt_image, bin_data, offset)[0]
        offset += struct.calcsize(fmt_image)
    return labels

## this part of code for data visualization by scattering plot
def visualizing(X, num_clusters, label, xlabel, ylabel, title):
    cmap = plt.get_cmap('hot')
    color_bar = [cmap(i) for i in np.linspace(0, 0.8, num_clusters)]
    for i in range(num_clusters):
        index = np.where(label == i)
        X_dig = X[index]
        plt.scatter(X_dig[:, 0], X_dig[:, 1], color=color_bar[i], label=str(i), s=10)
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()


## this part of code is for pca implementation, visualization and selection of eigenvectors
def pca_vis(img, num_comp=2):
    X = img.reshape(img.shape[0], -1)
    X_mean = X - np.mean(X, axis=0)
    cov_matrix = np.cov(X_mean, rowvar=False)

    # eigen-decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # plt.plot(list(range(len(eigenvalues))),sorted(eigenvalues))
    # plt.show()

    # ranking the eigenvalues and vectors
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]
    # sorted_eigenvalues = eigenvalues[sorted_indices]

    ##### This part of code is for cumulative variance visualization along the number of eigenvectors
    # total_variance = np.sum(sorted_eigenvalues)
    # cumulative_variance_ratio = np.cumsum(sorted_eigenvalues) / total_variance
    # plt.plot(list(range(len(cumulative_variance_ratio))),cumulative_variance_ratio, label="Represented Ratio", color="orange")
    # plt.xlabel("#PCs")
    # plt.ylabel("Variance%")
    # plt.fill_between(list(range(len(cumulative_variance_ratio))), cumulative_variance_ratio, alpha=0.2)
    # plt.grid(True,linestyle='--')
    # plt.title("The represented ratio along the number of PCs")
    # plt.show()

    # k = np.argmax(cumulative_variance_ratio >= percentage) + 1  # at least 80% variance
    # num = np.sum((sorted_eigenvalues > 1).astype(np.float32))


    W = sorted_eigenvectors[:,:num_comp] # using the first num_comp eigenvectors
    X_pca = X_mean @ W
    X_rebuild = (X_pca @ W.T + np.mean(X, axis=0)).reshape(img.shape)

    return X_pca, X_rebuild



## this part of code is for umap visualization
def umap_vis(img, n_neighbors=10, min_dist=0.3, n_components=2, random_state=42, spread=10):
    umap = UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components, random_state=random_state, spread=spread)
    X_umap = umap.fit_transform(img.reshape(img.shape[0], -1))
    visualizing(X_umap, 10, label,"UMAP-1", "UMAP-2", "UMAP visualization")


## draw the PCA for images of the same digit, and compare PCA difference
def pca_comp(img, label, digit, index):
    index = np.where(label == digit)
    img = img[index]
    new_img, img_pca = pca_vis(img)

    plt.scatter(new_img[:, 0], new_img[:, 1], color='red')
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.show()

    first_index, second_index = index[0], index[1]
    plt.figure(figsize=(10, 4))
    plt.tight_layout()
    plt.title("Comparison of images with large difference in 2D PCA space", fontsize=16)
    plt.axis("off")
    plt.subplot(1, 2, 1)
    plt.imshow(img[first_index])
    plt.subplot(1, 2, 2)
    plt.imshow(img[second_index])
    plt.show()

if __name__ == '__main__':

    img, label = decode_idx3_ubyte('data2_raw/train-images-idx3-ubyte'), decode_idx1_ubyte('data2_raw/train-labels-idx1-ubyte')
    umap_vis(img, n_neighbors=10, min_dist=0.8, n_components=2, spread=1)
    # basic setting neighbor=10, min_dist=0.3, spread=1



