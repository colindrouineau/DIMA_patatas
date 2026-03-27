from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.manifold import TSNE
import numpy as np
import matplotlib
import umap

import matplotlib.pyplot as plt

from format_data import DataFormatter
from patatas_code.data.viz_image import VizImage, COLORS
from data_processing import ProcessImage
from data_analysis import DataAnalyse


class Clustering:
    """class to recognise key components of a leaf"""

    def __init__(self, n_clusters, leaf):
        """initialises class instances useful for this class

        :param str leaf: name of the leaf which clustering will be applied on"""
        self.data_format = DataFormatter()
        self.visualise = VizImage()
        self.leaf = leaf
        self.n_clusters = n_clusters

    def load_data(self, channels=None):
        """Load all channels of the leaf pixels"""
        self.points, self.real_labels = self.data_format.leaf_mask_data(leaf=self.leaf)

    def cluster(self, X):
        """Runs kmeans clustering algorithm on X, with self.n_clusters,
        and prints silhouette score using kmeans labels, and then real labels"""
        self.kmeans = KMeans(n_clusters=self.n_clusters, init="k-means++")
        self.kmeans.fit(X)
        sil_score_kmeans = metrics.silhouette_score(X, self.kmeans.labels_)
        if len(X) == len(self.real_labels):
            sil_score_real = metrics.silhouette_score(X, self.real_labels)
            print(f"silhouette score using real labels is {sil_score_real}")
        print(f"silhouette score using clustering labels is {sil_score_kmeans}")

    def plot_clusters_on_leaf(
        self, labels=None, clustering_data="raw data", clustering_mthd="Kmeans"
    ):
        """plots on the leaf compared prediction distribution using both real labels
        and clustering labels.

        :param list[int] | None labels: Predicted labels. By default, kmeans labels"""
        if labels is None:
            labels = self.kmeans.labels_
        y_real, y_pred = self.data_format.reconstitute_leaf(self.leaf, labels)
        self.visualise.plot_y_real_pred(
            y_real, y_pred, title=f"{clustering_mthd} on {clustering_data}"
        )

    def load_tnse(self):
        """Compute tnse on the loaded points and saves it as an attribute"""
        tsne = TSNE()
        self.embedded_data = tsne.fit_transform(self.points)
        print("tnse transformed data shape : ", self.embedded_data.shape)

    def load_umap(self, n_neighbors=15, min_dist=0.1):
        # Create UMAP instance with default parameters
        reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist)
        # Fit and transform the data
        # possible to use gpu. Not necessary for 1 leaf, but may be interesting for a larger dataset.
        self.embedded_data = reducer.fit_transform(self.points, n_jobs=-1)

    def plot_embedded(
        self,
        labels=[0],
        dimred_mhtd="umap",
        clustering_data="raw data",
        clustering_mthd="Kmeans",
    ):
        """Plot 2D embedded data, colored using labels

        :param list[int] labels: labels used for coloring.
        By default, use kmeans labels saved as attributes"""
        x = self.embedded_data[:, 0]
        y = self.embedded_data[:, 1]
        label_type = "actual labels"
        if (
            len(labels) == 1
            and hasattr(self, "kmeans")
            and hasattr(self.kmeans, "labels_")
        ):
            labels = self.kmeans.labels_
            label_type = "cluster labels"
        labels = np.array(labels)  # for mask later

        dist = len(set(labels)) > 3
        if dist:
            cm = plt.get_cmap("viridis")
            colors = [cm(i) for i in np.linspace(0, 1, len(set(labels)))]

        # select each unique label and color matching points.
        for i, label in enumerate(list(set(labels))):
            # For each category, pick a color
            if dist:
                color = colors[i]
            else:
                color = COLORS[i % 10]

            mask = labels == label
            plt.scatter(x[mask], y[mask], color=color, s=1, marker="+")

        plt.title(
            f"{clustering_mthd} on {clustering_data} : viz on {dimred_mhtd} fit data. Labels = {label_type}"
        )
        plt.show()

    def transform_points(self):
        self.points = DataAnalyse().dataset_important(self.points)


if __name__ == "__main__":

    matplotlib.use("TkAgg")
    N_CLUSTERS = 2
    LEAF = "foliolo7_enves_a10"
    cluster = Clustering(n_clusters=N_CLUSTERS, leaf=LEAF)

    cluster.load_data()
    cluster.cluster(cluster.points)
    cluster.plot_clusters_on_leaf()

    cluster.load_tnse()
    cluster.plot_embedded()

    cluster.cluster(cluster.embedded_data)
    cluster.plot_embedded()
    cluster.plot_clusters_on_leaf()
