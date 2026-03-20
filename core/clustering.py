from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.manifold import TSNE
import numpy as np
import matplotlib

import matplotlib.pyplot as plt

from format_data import DataFormatter
from viz_image import VizImage


class Clustering:
    """class to recognise key components of a leaf"""

    def __init__(self, n_clusters, leaf):
        self.data_format = DataFormatter()
        self.visualise = VizImage()
        self.leaf = leaf
        self.n_clusters = n_clusters
        self.colors = np.array(
            [
                "#1f77b4",  # Blue
                "#ff7f0e",  # Orange
                "#2ca02c",  # Green
                "#d62728",  # Red
                "#9467bd",  # Purple
                "#8c564b",  # Brown
                "#e377c2",  # Pink
                "#7f7f7f",  # Gray
                "#bcbd22",  # Olive
                "#17becf",  # Cyan
            ]
        )

    def load_data(self, channels=None):
        self.points, _ = self.data_format.leaf_mask_data(leaf=self.leaf)

    def cluster(self, X):
        self.kmeans = KMeans(n_clusters=self.n_clusters, init="k-means++")
        self.kmeans.fit(X)
        sil_score = metrics.silhouette_score(X, self.kmeans.labels_)

        print(f"silhouette score is {sil_score}")

    def plot_clusters_on_leaf(self, labels=None):
        if labels is None:
            labels = self.kmeans.labels_
        y_real, y_pred = self.data_format.reconstitute_leaf(self.leaf, labels)
        self.visualise.plot_y_real_pred(y_real, y_pred)

    def load_tnse(self):
        tsne = TSNE()
        self.embedded_data = tsne.fit_transform(self.points)
        print("tnse transformed data shape : ", self.embedded_data.shape)

    def plot_tnse(self, labels=[0]):
        x = self.embedded_data[:, 0]
        y = self.embedded_data[:, 1]
        if (
            len(labels) == 1
            and hasattr(self, "kmeans")
            and hasattr(self.kmeans, "labels_")
        ):
            labels = self.kmeans.labels_
        labels = np.array(labels)  # for mask later

        for i, label in enumerate(list(set(labels))):
            # For each category, pick a color and marker
            color = self.colors[i % 10]
            mask = labels == label
            plt.scatter(x[mask], y[mask], color=color, s=10)

        plt.title("Clustering on tsne fit data")
        plt.show()


if __name__ == "__main__":

    matplotlib.use("TkAgg")
    N_CLUSTERS = 2
    LEAF = "foliolo7_enves_a10"
    cluster = Clustering(n_clusters=N_CLUSTERS, leaf=LEAF)

    cluster.load_data()
    cluster.cluster(cluster.points)
    cluster.plot_clusters_on_leaf()

    cluster.load_tnse()
    cluster.plot_tnse()

    cluster.cluster(cluster.embedded_data)
    cluster.plot_tnse()
    cluster.plot_clusters_on_leaf()
