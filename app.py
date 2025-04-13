import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import pandas as pd

# Set page configuration
st.set_page_config(page_title="Fashion MNIST Clustering", layout="wide")

# Title and description
st.title("Fashion MNIST Clustering Analysis")
st.markdown("""
This app performs K-means clustering on the Fashion MNIST dataset using both random and class-based initialization.
It visualizes sample images, cluster centers, and clustering results.
""")

@st.cache_data
def load_and_prepare_data():
    # Load and prepare Fashion MNIST dataset
    fashion_mnist = fetch_openml(name='Fashion-MNIST', version=1, parser='auto')
    X = fashion_mnist.data.astype('float32') / 255.0
    y = fashion_mnist.target.astype('int')
    
    # Use a subset for faster computation
    X_sample, _, y_sample, _ = train_test_split(X, y, test_size=0.8, random_state=42)
    return X, y, X_sample, y_sample

# Load data
X, y, X_sample, y_sample = load_and_prepare_data()

# Display dataset information
st.header("Dataset Information")
st.write(f"Dataset shape: {X.shape}")
st.write(f"Number of classes: {len(np.unique(y))}")
st.write(f"Class distribution: {np.bincount(y)}")
st.write(f"Working with sample of shape: {X_sample.shape}")

# Visualize sample images
st.header("Sample Images")
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
for i, ax in enumerate(axes.flat):
    ax.imshow(X.iloc[i].values.reshape(28, 28), cmap='gray')
    ax.set_title(f"Class: {y.iloc[i]}")
    ax.axis('off')
plt.tight_layout()
st.pyplot(fig)

# K-means clustering with random initialization
st.header("K-means Clustering Results")

# Random initialization
np.random.seed(42)
min_vals = X_sample.min(axis=0)
max_vals = X_sample.max(axis=0)
random_init = np.random.uniform(min_vals, max_vals, size=(10, 784))

kmeans_random = KMeans(n_clusters=10, init=random_init, n_init=1, random_state=42)
clusters_random = kmeans_random.fit_predict(X_sample)

# Class-based initialization
class_based_init = np.zeros((10, 784))
for class_idx in range(10):
    class_samples = X[y == class_idx]
    class_based_init[class_idx] = class_samples.iloc[0].values

kmeans_class = KMeans(n_clusters=10, init=class_based_init, n_init=1, random_state=42)
clusters_class = kmeans_class.fit_predict(X_sample)

# Display cluster counts
col1, col2 = st.columns(2)
with col1:
    st.subheader("Random Initialization Cluster Counts")
    cluster_counts_random = np.bincount(clusters_random)
    for i, count in enumerate(cluster_counts_random):
        st.write(f"Cluster {i}: {count} points")

with col2:
    st.subheader("Class-Based Initialization Cluster Counts")
    cluster_counts_class = np.bincount(clusters_class)
    for i, count in enumerate(cluster_counts_class):
        st.write(f"Cluster {i}: {count} points")

# Visualize cluster centers
st.subheader("Cluster Centers")
tab1, tab2 = st.tabs(["Random Initialization", "Class-Based Initialization"])

with tab1:
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    for i, ax in enumerate(axes.flat):
        ax.imshow(kmeans_random.cluster_centers_[i].reshape(28, 28), cmap='viridis')
        ax.set_title(f"Cluster {i}")
        ax.axis('off')
    plt.tight_layout()
    st.pyplot(fig)

with tab2:
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    for i, ax in enumerate(axes.flat):
        ax.imshow(kmeans_class.cluster_centers_[i].reshape(28, 28), cmap='viridis')
        ax.set_title(f"Cluster {i}")
        ax.axis('off')
    plt.tight_layout()
    st.pyplot(fig)

# Visualize sample images from clusters
st.subheader("Sample Images from Each Cluster")
tab1, tab2 = st.tabs(["Random Initialization", "Class-Based Initialization"])

with tab1:
    fig, axes = plt.subplots(10, 10, figsize=(15, 20))
    for cluster_idx in range(10):
        cluster_samples = X_sample[clusters_random == cluster_idx]
        for i in range(10):
            if i < len(cluster_samples):
                axes[cluster_idx, i].imshow(cluster_samples.iloc[i].values.reshape(28, 28), cmap='gray')
                axes[cluster_idx, i].axis('off')
    plt.tight_layout()
    st.pyplot(fig)

with tab2:
    fig, axes = plt.subplots(10, 10, figsize=(15, 20))
    for cluster_idx in range(10):
        cluster_samples = X_sample[clusters_class == cluster_idx]
        for i in range(10):
            if i < len(cluster_samples):
                axes[cluster_idx, i].imshow(cluster_samples.iloc[i].values.reshape(28, 28), cmap='gray')
                axes[cluster_idx, i].axis('off')
    plt.tight_layout()
    st.pyplot(fig)

# SSE comparison
st.header("Clustering Quality Comparison")
sse_random = kmeans_random.inertia_
sse_class = kmeans_class.inertia_

fig, ax = plt.subplots(figsize=(8, 6))
ax.bar(['Random Initialization', 'Class-Based Initialization'], [sse_random, sse_class])
ax.set_ylabel('Sum of Squared Error (SSE)')
ax.set_title('Clustering Quality Comparison')
ax.set_ylim(0, max(sse_random, sse_class) * 1.1)
for i, v in enumerate([sse_random, sse_class]):
    ax.text(i, v + 100, f"{v:.2f}", ha='center')
plt.tight_layout()
st.pyplot(fig)

# Confusion matrix
st.header("Cluster vs. Class Distribution")
cluster_class_matrix = np.zeros((10, 10), dtype=int)
for i in range(len(y_sample)):
    cluster_class_matrix[clusters_class[i], y_sample.iloc[i]] += 1

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(cluster_class_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
ax.set_xlabel('True Class')
ax.set_ylabel('Cluster')
ax.set_title('Cluster vs. Class Distribution')
plt.tight_layout()
st.pyplot(fig)
