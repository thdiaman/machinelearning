import numpy as np
import pandas as pd
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

# Read data
pdata = pd.read_csv("pdata.csv", sep=';')

# Plot data
fig = plt.figure()
ax = Axes3D(fig)
for index, row in pdata.iterrows():
    ax.scatter(row['X'], row['Y'], row['Z'], color='b') 
    ax.text(row['X'], row['Y'], row['Z'], " x" + str(index + 1))
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Apply principal component analysis
pca = PCA(n_components=2)
pca.fit(pdata)
print(pca.explained_variance_ratio_)

# Transform the data
pdata_transformed = pca.transform(pdata)
pdata_transformed = pd.DataFrame(data={'PC1': pdata_transformed[:, 0], 'PC2': pdata_transformed[:, 1]})

# Plot data
fig, ax = plt.subplots()
for index, row in pdata_transformed.iterrows():
    ax.scatter(row['PC1'], row['PC2'], color='b') 
    ax.text(row['PC1'], row['PC2'], " x" + str(index + 1))
ax.set_xlabel('PC1')
ax.set_xlabel('PC2')
plt.show()
