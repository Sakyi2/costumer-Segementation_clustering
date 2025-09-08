# utils.py
"""
Utility functions for Customer Segmentation Project
Author: Your Name
"""

# ===============================
# Imports
# ===============================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn import metrics
from yellowbrick.cluster import KElbowVisualizer
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap


# ===============================
# Data Loading & Preprocessing
# ===============================
def load_data(filepath: str) -> pd.DataFrame:
    """Load dataset from a CSV file."""
    return pd.read_csv(filepath)


def encode_labels(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Label-encode categorical columns."""
    encoder = LabelEncoder()
    for col in cols:
        df[col] = encoder.fit_transform(df[col])
    return df


def scale_data(df: pd.DataFrame, cols_to_scale: list) -> pd.DataFrame:
    """Standardize numeric columns."""
    scaler = StandardScaler()
    df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
    return df


# ===============================
# PCA (Dimensionality Reduction)
# ===============================
def apply_pca(df: pd.DataFrame, n_components: int = 2):
    """Apply PCA and return transformed data and fitted PCA object."""
    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(df)
    return reduced, pca


def plot_pca_variance(pca):
    """Plot cumulative explained variance ratio."""
    plt.figure(figsize=(6, 4))
    plt.plot(np.cumsum(pca.explained_variance_ratio_), marker="o")
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title("PCA Explained Variance")
    plt.grid()
    plt.show()


# ===============================
# Clustering (KMeans + Agglomerative)
# ===============================
def run_kmeans(data, n_clusters: int = 3, random_state: int = 42):
    """Fit KMeans and return labels and model."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    labels = kmeans.fit_predict(data)
    return labels, kmeans


def run_agglomerative(data, n_clusters: int = 3):
    """Fit Agglomerative Clustering and return labels."""
    agg = AgglomerativeClustering(n_clusters=n_clusters)
    labels = agg.fit_predict(data)
    return labels, agg


def describe_clusters(df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Attach cluster labels and return summary statistics."""
    df["Cluster"] = labels
    return df.groupby("Cluster").mean()


# ===============================
# Visualization Helpers
# ===============================
def plot_clusters_2d(data, labels):
    """Plot clusters in 2D space."""
    plt.figure(figsize=(6, 6))
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap="viridis", s=50)
    plt.title("Customer Segments (2D)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.show()


def plot_clusters_3d(data, labels):
    """Plot clusters in 3D space."""
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=labels, cmap="viridis", s=50)
    ax.set_title("Customer Segments (3D)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    plt.show()
