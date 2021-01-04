import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn import datasets
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

seed = 1997
np.random.seed(seed)

data_set = datasets.fetch_olivetti_faces()
data = data_set.data
target = data_set.target

# for i in np.arange(0, 350, 8):
#     plt.imshow(data_set.images[i])
#     plt.show()

strat_split_train_test = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
train_ind, test_ind = next(strat_split_train_test.split(data,target))

train_data = data[train_ind]
train_target = target[train_ind]
test_data = data[test_ind]
test_target = target[test_ind]


strat_split_train_valid = StratifiedShuffleSplit(n_splits=1, test_size=0.33, random_state=seed)
train_ind, valid_ind = next(strat_split_train_test.split(train_data,train_target))

valid_data = train_data[valid_ind]
valid_target = train_target[valid_ind]
train_data = train_data[train_ind]
train_target = train_target[train_ind]


print('train_data size : ', len(train_data))
print('valid_data size : ', len(valid_data))
print('test_data size : ',len(test_data))

silhouette_scores = []
n_clusters = []
models = []
# for n in range(10, 200, 10):
#     k_means = KMeans(n_clusters = n, random_state=seed).fit(train_data)
#     n_clusters.append(n)
#     silhouette_scores.append(silhouette_score(train_data, k_means.labels_))
#     print(n)
#
# best_model = models[np.argmax(silhouette_scores)]

best_model = KMeans(n_clusters = 100, random_state=seed).fit(train_data)
def plot_faces(faces, labels, n_cols=5):
    n_rows = (len(faces) - 1) // n_cols + 1
    plt.figure(figsize=(n_cols, n_rows * 1.1))
    for index, (face, label) in enumerate(zip(faces, labels)):
        plt.subplot(n_rows, n_cols, index + 1)
        plt.imshow(face.reshape(64, 64), cmap="gray")
        plt.axis("off")
        plt.title(label)


for cluster_id in np.unique(best_model.labels_):
    print("Cluster", cluster_id)
    in_cluster = best_model.labels_==cluster_id
    faces = train_data[in_cluster].reshape(-1, 64, 64)
    labels = train_target[in_cluster]
    plot_faces(faces, labels)
plt.show()
print('moma')




