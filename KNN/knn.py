import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split  
from sklearn.metrics import classification_report, confusion_matrix

# Read the data
data = pd.read_csv("salary.csv", sep=';')
X = data.iloc[:, 0:-1]
y = data.iloc[:, -1]

# Convert problem to classification
y = np.digitize(y, [1500])
data.iloc[:, -1] = y

# Plot the data
fig, ax = plt.subplots()
groups = data.groupby('Salary')
for name, group in groups:
    ax.plot(group.iloc[:, 0], group.iloc[:, 1], marker='o', linestyle='', label=name)
ax.set_xlabel(data.columns.values[0])
ax.set_ylabel(data.columns.values[1])
ax.legend()
plt.show()

# Split to training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Train the model
model = KNeighborsClassifier(n_neighbors=4)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)  
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
