import numpy as np
import pandas as pd
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split  
from sklearn.preprocessing import MinMaxScaler

# Read data
pdata = pd.read_csv("wine.csv", sep=';')

# Split the data
trainingdata, testdata = train_test_split(pdata, test_size=0.30, random_state=2)
X_train, Y_train = trainingdata.iloc[:, 0:-1], trainingdata.iloc[:, -1]
X_test, Y_test = testdata.iloc[:, 0:-1], testdata.iloc[:, -1]

# Scale the data
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Apply classifier
model = KNeighborsClassifier(n_neighbors=4)
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)
print(classification_report(Y_test, Y_pred))

# Apply PCA
pca = PCA(n_components=X_train.shape[1])
pca.fit(X_train)

# Plot variance ratio
variance = pca.explained_variance_ratio_
xticks = list(range(len(variance)))
plt.bar(xticks, variance)
plt.xticks(xticks, ['PC' + str(i+1) for i in xticks])
plt.xlabel('Principal Components')
plt.ylabel('Percentage of Variance')
plt.show()

# Transform data
X_train_transformed = pca.transform(X_train)
X_test_transformed = pca.transform(X_test)

# Apply classifier using first 4 PCs
PCs = 4
model = KNeighborsClassifier(n_neighbors=4)
model.fit(X_train_transformed[:, :PCs], Y_train)
Y_pred = model.predict(X_test_transformed[:, :PCs])
print(classification_report(Y_test, Y_pred))
