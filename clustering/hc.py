import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('Mall_Customers.csv') 

X = df.iloc[:,[2,3]].to_numpy()

import scipy.cluster.hierarchy as sch

dendrogram = sch.dendrogram(sch.linkage(X, method="ward"))
plt.title("Dendrogram")
plt.xlabel("Points")
plt.ylabel("Distance")
plt.show()

from sklearn.cluster import AgglomerativeClustering

cluster = AgglomerativeClustering(n_clusters=2, metric="euclidean", linkage="ward")
y_cluster = cluster.fit_predict(X)
plt.scatter(X[y_cluster == 0, 0], X[y_cluster == 0, 1], c='red', label = "Cluster 1")
plt.scatter(X[y_cluster == 1, 0], X[y_cluster == 1, 1], c='blue', label = "Cluster 2")
plt.title("HC")
plt.xlabel("Age")
plt.ylabel("Annual Income")
plt.legend()
plt.show()