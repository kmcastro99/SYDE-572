"""
SYDE 572 - Assignment 3

Author: Karla Castro

Question 2

"""
from torchvision.datasets import MNIST
from sklearn.decomposition import PCA
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans # To check if implementation works

# Calculate the euclidean distance of two points (Taken from previous assignment)
def euclidean_distance(x, y):
    """
    Calculates the euclidean distanc of two points
    """
    distance = np.linalg.norm(x - y, axis=2)
    return distance

# Define the K-Means function
def k_means(x, k, max_iterations=100):
    """
    Applies K-Means clustering to the input data.
    Parameters:
        - x: Input data (numpy array), observations we want to cluster
        - k: Number of clusters
        - max_iters: Maximum number of iterations to run the algorithm. 
                     To avoid infinite loops
        Returns:
        - clusters: List of clusters
        - centroids: Cluster centers
        """
    # Initialize centroids by randomly selecting k data points from the input data
    initial_centroid_indices = np.random.choice(x.shape[0], k, replace=False)
    # Get the data points at the indices
    centroids = x[initial_centroid_indices]
    # Initialize the cluster labels to -1
    i = 0
    # Initialize convergence flag for while loop
    convergence = False

    # Loop until convergence or max iterations reached
    while i < max_iterations and not convergence:
        # Calculate distances between data points and centroids
        distances = euclidean_distance(x[:, np.newaxis], centroids)
        
        # Assign each data point to the nearest centroid
        labels = np.argmin(distances, axis=1)
        
        # Update centroids as the mean of data points in each cluster
        new_centroids = np.array([x[labels == i].mean(axis=0) for i in range(k)])# Check for convergence (if the centroids have changed)

        if np.array_equal(new_centroids, centroids):
            # Set convergence flag to True to stop algorithm
            convergence = True
        # Update centroids
        centroids = new_centroids
        i += 1

    return labels, centroids

# Calculate cluster consistency
def cluster_consistency(cluster_labels, k):
    """
    Calculates the cluster consistency of a clustering result.
    Parameters:
        - cluster_labels: Cluster labels
        - k: Number of clusters
    Returns:
        - cluster_consistency: Cluster consistency
    """
    # Initialize an array to store the cluster consistency for each cluster
    cluster_consistencies = []
    # Loop through each cluster
    for i in range(k):
        # Get the indices of the data points in the cluster
        cluster_indices = np.where(cluster_labels == i)[0]
        cluster_classes = np.zeros(10)  # Initialize an array to count class occurrences (assuming 10 classes)
        # Loop through each data point in the cluster
        for index in cluster_indices:
            # Get the class label of the data point
            class_label = int(train_labels_flat[index])
            cluster_classes[class_label] += 1

        # Get the most common class count
        mi = np.max(cluster_classes)
        # Use len to find the total number of points in cluster
        Ni = len(cluster_indices)
        # Overall cluster consistency
        Qi = mi / Ni  # Cluster consistency for i
        cluster_consistencies.append(Qi)

    # Overall clustering consistency Q
    Q_consistency = np.mean(cluster_consistencies)
    return Q_consistency

# Function to plot
def plot_clusters(x, cluster_labels, k):
    """
    Plots the clusters in 2D.
    Parameters:
        - x: Input data (numpy array), observations we want to cluster
        - cluster_labels: Cluster labels
        - k: Number of clusters
    """
    colors = plt.cm.rainbow(np.linspace(0, 1, k))
    for i in range(k):
        cluster_data = x[cluster_labels == i]
        plt.scatter(cluster_data[:, 0], cluster_data[:, 1], c=colors[i], label=f'Cluster {i+1}')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.title(f'K-Means Clustering with {k} Clusters')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.show()

# Using kmeans library
def kmeans_clustering(x, k, max_iterations=100):
    """
    Perform K-means clustering using the scikit learn
    Returns:
        - cluster_labels: Cluster labels
        - centroids: Cluster centers
    """
    kmeans = KMeans(n_clusters=k, max_iter=max_iterations, random_state=0)
    cluster_labels = kmeans.fit_predict(x)
    cluster_centers = kmeans.cluster_centers_
    
    return cluster_labels, cluster_centers
# ---------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------

# Define seed
random.seed(0)
# Define the number of clusters (k)
k_values = [5, 10, 20, 40]

# Load the MNIST dataset (only 100 data points for each class)
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

# Apply K-Means clustering for k value = 5
cluster_k1, centroids_k1 = k_means(train_pca, k_values[0])
# Plot 
plot_clusters(train_pca, cluster_k1, k_values[0])

# Apply K-Means clustering for k value = 10
cluster_k2, centroids_k2 = k_means(train_pca, k_values[1])
# Plot
plot_clusters(train_pca, cluster_k2, k_values[1])

# Apply K-Means clustering for k value = 20
cluster_k3, centroids_k3 = k_means(train_pca, k_values[2])
#Plot
plot_clusters(train_pca, cluster_k3, k_values[2])

# Apply K-Means clustering for k value = 40
cluster_k4, centroids_k4 = k_means(train_pca, k_values[3])
# Plot
plot_clusters(train_pca, cluster_k4, k_values[3])

# Calculate Consistency

cluster_consistency_k1 = cluster_consistency(cluster_k1, k_values[0])
cluster_consistency_k2 = cluster_consistency(cluster_k2, k_values[1])
cluster_consistency_k3 = cluster_consistency(cluster_k3, k_values[2])
cluster_consistency_k4 = cluster_consistency(cluster_k4, k_values[3])

# Print consistency
print ('Consistency using own K-mean function')
print(f'The consistency for k= 5 is:{cluster_consistency_k1}')
print(f'The consistency for k= 10 is:{cluster_consistency_k2}')
print(f'The consistency for k= 20 is:{cluster_consistency_k3}')
print(f'The consistency for k= 40 is:{cluster_consistency_k4}')

# Apply K-Means clustering (sklearn) for k value = 5
cluster_k1_2, centroids_k1_2 = kmeans_clustering(train_pca, k_values[0])
# Compute consistency
cluster_consistency_k1_2 = cluster_consistency(cluster_k1_2, k_values[0])

# Apply K-Means clustering (sklearn) for k value = 10
cluster_k2_2, centroids_k2_2 = kmeans_clustering(train_pca, k_values[1])
# Compute consistency
cluster_consistency_k2_2 = cluster_consistency(cluster_k2_2, k_values[1])

# Apply K-Means clustering (sklearn) for k value = 20
cluster_k3_2, centroids_k3_2 = kmeans_clustering(train_pca, k_values[2])
# Compute consistency
cluster_consistency_k3_2 = cluster_consistency(cluster_k3_2, k_values[2])

# Apply K-Means clustering (sklearn) for k value = 40
cluster_k4_2, centroids_k4_2 = kmeans_clustering(train_pca, k_values[3])
# Compute consistency
cluster_consistency_k4_2 = cluster_consistency(cluster_k4_2, k_values[3])

# Print consistency
print ('Consistency using KMeans library from sklearn')
print(f'The consistency for k= 5 is:{cluster_consistency_k1_2}')
print(f'The consistency for k= 10 is:{cluster_consistency_k2_2}')
print(f'The consistency for k= 20 is:{cluster_consistency_k3_2}')
print(f'The consistency for k= 40 is:{cluster_consistency_k4_2}')
