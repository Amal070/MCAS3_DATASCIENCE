import pandas as pd
customers = pd.read_csv('customer_data.csv')
print(customers.head())
#Visualize data Annual Income vs spending Score
import matplotlib.pyplot as plt
points = customers.iloc[:, 3:5].values
x = points[:, 0]
y = points[:, 1]
plt.scatter(x, y, s=50, alpha=0.7)
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score')
plt.show()
#Using k means clustering create 6 clusters of customers based on their spending pattern.Visualize the same in a scatter plot with each cluster in a different color scheme.
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=6, random_state=0)
kmeans.fit(points)
predicted_cluster_indexes = kmeans.predict(points)
plt.scatter(x, y, c=predicted_cluster_indexes, s=50, alpha=0.7, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=100)
plt.show()
#print cluster index for each data point after applying K Means
print("\nPredicted Cluster Indexes:\n",predicted_cluster_indexes)
