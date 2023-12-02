import numpy as np
import pandas as pd
from sklearn.metrics import euclidean_distances
import matplotlib.pyplot as plt

def kmeans(X, k, max_iters=100):
    # Randomly initialize centroids
    centroids = X[np.random.choice(len(X), k, replace=False)]

    for _ in range(max_iters):
        # Assign each data point to the nearest centroid using sklearn's euclidean_distances
        distances = euclidean_distances(X, centroids)
        labels = np.argmin(distances, axis=1)

        # Update centroids based on the mean of data points in each cluster
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])

        # Check for convergence
        if np.all(centroids == new_centroids):
            break

        centroids = new_centroids

    return centroids, labels

# Read data from CSV file
df = pd.read_csv('DataAnalyticsLab\KMeans\kmeans.csv')  # Replace 'your_data.csv' with your actual file name

# Extract features from the DataFrame
X = df.values

# Specify the number of clusters (k)
k = 3

# Apply k-means algorithm
centroids , labels = kmeans(X, k)

# Plot the data points and cluster centroids
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50, alpha=0.8)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200, label='Centroids')
plt.title('K-Means Clustering')
plt.legend()
plt.show()
