from sklearn.datasets import make_blobs
import hdbscan
import matplotlib.pyplot as plt

N_SAMPLES = 1000
RANDOM_STATE = 42
X, y = make_blobs(n_samples=N_SAMPLES,
                  cluster_std=[2.0, 0.5],
                  centers=[(0, 0), (5, 5)],
                  random_state=RANDOM_STATE)

plt.figure(figsize = (10, 10))
plt.scatter(X[:, 0], X[:, 1])
plt.show()

hclusterer = hdbscan.HDBSCAN(min_cluster_size=5).fit(X)

plt.figure(figsize = (10, 10))
plt.scatter(X[:, 0], X[:, 1], c = hclusterer.labels_)
plt.show()

