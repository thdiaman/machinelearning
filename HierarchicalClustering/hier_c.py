import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

# Read the data
data = pd.read_csv("europe.csv", sep=';')
countries = data.iloc[:, 0].tolist()
X = data.iloc[:, 1:]

# Scale the data
X = MinMaxScaler().fit_transform(X)

# Apply hierarchical clustering
Z = linkage(X, method='complete')

# Plot the dendrogram given a threshold
threshold = 0.4 # alternatively use the number of clusters, i.e. k = 6, and then threshold = Z[-(k-1), 2]
fig, ax = plt.subplots()
dendrogram(Z, labels = countries, color_threshold = threshold, leaf_rotation = 90)
ax.axhline(y = threshold, c = 'k')
fig.subplots_adjust(bottom=0.35)

# Print the labels given the same threshold
labels = fcluster(Z, threshold, criterion='distance') # or fcluster(Z, k, criterion='maxclust')
print(labels)

plt.show()
