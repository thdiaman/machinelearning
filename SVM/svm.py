import numpy as np
import pandas as pd
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score

# Read the data
data = pd.read_csv("data.csv", sep=';')
X = data.iloc[:, 0:2]
y = data.iloc[:, 2]

# Split to training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
traindata = pd.DataFrame(data=np.c_[X_train, y_train], columns=['V0', 'V1', 'target'])

# Plot data
fig, ax = plt.subplots()
groups = traindata.groupby('target')
for name, group in groups:
    ax.plot(group.iloc[:, 0], group.iloc[:, 1], marker='o', linestyle='', label=name)
ax.set_xlabel('V0')
ax.set_ylabel('V1')
ax.legend()

# Create meshgrid
x0_min, x0_max = X.iloc[:, 0].min() - 1, X.iloc[:, 0].max() + 1
x1_min, x1_max = X.iloc[:, 1].min() - 1, X.iloc[:, 1].max() + 1
x0, x1 = np.meshgrid(np.arange(x0_min, x0_max, 0.02), np.arange(x1_min, x1_max, 0.02))

# RBF kernel with gamma 1
model = SVC(kernel='rbf', gamma=1, C=1.0)
model.fit(X_train, y_train)
Z = model.predict(np.c_[x0.ravel(), x1.ravel()])
Z = Z.reshape(x0.shape)
CS = ax.contour(x0, x1, Z, colors=['blue'])
labels = ax.clabel(CS, fmt="gamma=1")

# RBF kernel with gamma 0.01
model = SVC(kernel='rbf', gamma=0.01, C=1.0)
model.fit(X_train, y_train)
Z = model.predict(np.c_[x0.ravel(), x1.ravel()])
Z = Z.reshape(x0.shape)
CS = ax.contour(x0, x1, Z, colors=['red'])
labels = ax.clabel(CS, fmt="gamma=0.01")

# RBF kernel with gamma 100
model = SVC(kernel='rbf', gamma=100, C=1.0)
model.fit(X_train, y_train)
Z = model.predict(np.c_[x0.ravel(), x1.ravel()])
Z = Z.reshape(x0.shape)
CS = ax.contour(x0, x1, Z, colors=['gray'])
labels = ax.clabel(CS, fmt="gamma=100")

# Find training and testing error
gammavalues = [0.01, 0.1, 1, 10, 100, 1000, 10000, 100000, 1000000]
trainingerror, testingerror = [], []
for gamma in gammavalues:
    model = SVC(kernel='rbf', gamma=gamma, C=1.0)
    model.fit(X_train, y_train)
    trainingerror.append(1 - accuracy_score(y_train, model.predict(X_train)))
    testingerror.append(1 - accuracy_score(y_test, model.predict(X_test)))

# Plot training and testing error
fig, ax = plt.subplots()
ax.plot(trainingerror, label="Training Error")
ax.plot(testingerror, label="Testing Error")
ax.set_xticks(range(len(gammavalues)))
ax.set_xticklabels(gammavalues)
ax.set_xlabel("gamma")
ax.legend()

# Find best gamma using cross validation
accuracies = []
for gamma in gammavalues:
    model = SVC(kernel='rbf', gamma=gamma, C=1.0)
    scores = cross_val_score(model, X_train, y_train, cv=10)
    accuracies.append(np.mean(scores))

# Plot accuracy vs gamma
fig, ax = plt.subplots()
ax.plot(accuracies)
ax.set_xticks(range(len(gammavalues)))
ax.set_xticklabels(gammavalues)
ax.set_xlabel("gamma")
ax.set_ylabel("Accuracy")

plt.show()

