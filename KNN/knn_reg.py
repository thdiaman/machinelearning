import numpy as np
import pandas as pd
from math import sqrt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split  
from sklearn.metrics import mean_squared_error, r2_score

# Read the data
data = pd.read_csv("salary.csv", sep=';')
X = data.iloc[:, 0:-1]
y = data.iloc[:, -1]

# Plot the data
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(X.iloc[:, 0], X.iloc[:, 1], y)
ax.set_xlabel(data.columns.values[0])
ax.set_ylabel(data.columns.values[1])
ax.set_zlabel(data.columns.values[2])
plt.show()

# Split to training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Train the model
model = KNeighborsRegressor(n_neighbors=4)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)  
print(sqrt(mean_squared_error(y_test, y_pred)))
print(r2_score(y_test, y_pred))
