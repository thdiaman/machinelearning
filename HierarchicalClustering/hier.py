import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

# Read the data
data = pd.read_csv("data.csv", sep=';')
indexes = ["x" + str(i + 1) for i in range(len(data))]

# Plot the data with labels
fig, ax = plt.subplots()
for index, row in data.iterrows():
    ax.scatter(row['X'], row['Y'], color='b')
    ax.text(row['X'], row['Y'], " x" + str(index + 1))
ax.set_xlabel('X')
ax.set_ylabel('Y')

# Apply hierarchical clustering
Z = linkage(data, method='single') # or complete

# Plot the dendrogram
fig, ax = plt.subplots()
dendrogram(Z, labels = indexes, color_threshold=100)

# Get the labels for 3 clusters
labels = fcluster(Z, 3, criterion='maxclust')
print(labels)

# Plot the data with labels
fig, ax = plt.subplots()
for index, row in data.iterrows():
    ax.scatter(row['X'], row['Y'], color=['b', 'r', 'g'][labels[index] - 1])
    ax.text(row['X'], row['Y'], " x" + str(index + 1))
ax.set_xlabel('X')
ax.set_ylabel('Y')
plt.show()  

