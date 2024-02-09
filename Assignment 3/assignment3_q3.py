"""
SYDE 572 - Assignment 3

Author: Karla Castro

Question 3

"""
from torchvision.datasets import MNIST
from sklearn.decomposition import PCA
import numpy as np
import random
import matplotlib.pyplot as plt

def expectation_maximization(x, num_clusters , max_iterations = 500):
    """
    Perform EM for GMM (to estimate mean, variance, and weights)
    Parameters:
    - data: data to cluster
    - num_clusters: number of clusters
    - iterations: number of iterations to run EM
    Returns:
    - cluster_means: means of clusters
    - cluster_variances: variances of clusters
    - cluster_weights: weights of clusters
    """
    # Initialize parameters
    num_data_points, data_dimension = x.shape
    # Initialize cluster means and covariances
    cluster_means = x[np.random.choice(num_data_points, num_clusters, replace=False)]
    cluster_covariances = np.tile(np.identity(data_dimension), (num_clusters, 1, 1))
    cluster_weights = np.ones(num_clusters) / num_clusters

    for i in range(max_iterations):
        # E-step
        resp = np.zeros((num_data_points, num_clusters))
        # Calculate the responsibilities
        for j in range(num_clusters):
            # Calculate the mahalanobis distance
            diff = x[:, np.newaxis] - cluster_means[j]
            inv_cov = np.linalg.inv(cluster_covariances[j])
            mahalanobis = np.sum(diff @ inv_cov * diff, axis=2)
            # Calculate the responsibilities
            for k in range(num_data_points):
                resp[k, j] = np.exp(-0.5 * mahalanobis[k]) / (
                np.sqrt(2 * np.pi * np.linalg.det(cluster_covariances[j])) * cluster_weights[j])
        resp /= resp.sum(axis=1, keepdims=True)
        # Maximization
        N = np.sum(resp, axis=0)
        for i in range(num_clusters):
            # Update the cluster means, covariances, and weights
            cluster_means[i] = np.sum(x * resp[:, i, np.newaxis], axis=0) / N[i]
            diff = x - cluster_means[i]
            cluster_covariances[i] = np.dot((resp[:, i, np.newaxis] * diff).T, diff) / N[i]
            cluster_weights[i] = N[i] / num_data_points

    return cluster_means, cluster_covariances, cluster_weights, resp

# ----------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------

# Test

# Define seed
random.seed(0)
# Define the number of clusters (k)
k_value = 5

# Load the MNIST dataset
dataset = MNIST(root='data', train=True, download=True)
train_images = dataset.data.numpy()
train_labels = dataset.targets.numpy()

 # Flatten images
train_images_flat = train_images.reshape(-1, 28*28)
train_labels_flat = train_labels.reshape(-1, 1)

# Perform PCA with 2 components
pca = PCA(n_components=2)
train_pca = pca.fit_transform(train_images_flat).astype(np.float32)
train_labels_flat = train_labels_flat.astype(np.float32)

# Usee all the points from the dataset
np.random.shuffle(train_pca)
train_pca = train_pca[:]
np.random.shuffle(train_labels_flat)
train_labels_flat = train_labels_flat[:]

# Apply EM for k value = 5
cluster_means_k1, cluster_variances_k1, cluster_weights_k1, resp= expectation_maximization(train_pca, k_value)
# Plot

