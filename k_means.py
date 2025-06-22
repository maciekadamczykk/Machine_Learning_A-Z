import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("Mall_Customers.csv") 

X = df.iloc[:, 1:] 

from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(sparse_output=False)  
genre_encoded = encoder.fit_transform(X[["Genre"]])

genre_df = pd.DataFrame(genre_encoded, columns=encoder.get_feature_names_out(["Genre"]))

X = pd.concat([genre_df, X.drop("Genre", axis=1)], axis=1)

print(X.head())

import seaborn as sns
corr_matrix = X.corr()
sns.heatmap(corr_matrix,annot=True)
plt.show()

from sklearn.cluster import KMeans

wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i, init = "k-means++", random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11), wcss)
plt.title("The elbow method")
plt.xlabel("Number of clusters")
plt.ylabel("WCSS")
plt.show()



