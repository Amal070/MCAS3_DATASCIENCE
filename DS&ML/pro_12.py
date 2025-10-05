import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Step 1: Load dataset
customers = pd.read_csv('customer_data.csv')
print(customers.head())

# Step 2: Visualize Annual Income vs Spending Score
points = customers.iloc[:, 3:5].values  # Selecting columns: Annual Income & Spending Score
x = points[:, 0]
y = points[:, 1]

plt.figure(figsize=(8,5))
plt.scatter(x, y, s=50, alpha=0.7)
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score')
plt.title('Customer Spending Patterns')
plt.show()

# Step 3: Apply K-Means clustering (6 clusters)
kmeans = KMeans(n_clusters=6, random_state=0)
kmeans.fit(points)
predicted_cluster_indexes = kmeans.predict(points)

# Step 4: Visualize clusters
plt.figure(figsize=(8,5))
plt.scatter(x, y, c=predicted_cluster_indexes, s=50, alpha=0.7, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=100, marker='X', label='Centroids')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score')
plt.title('Customer Segmentation using K-Means')
plt.legend()
plt.show()

# Step 5: Print cluster index for each data point
print("\nPredicted Cluster Indexes:\n", predicted_cluster_indexes)
