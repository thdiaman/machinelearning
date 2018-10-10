import pydotplus
import numpy as np
import pandas as pd
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split  
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import classification_report, confusion_matrix

# Read the data
data = pd.read_csv("weatherfull.csv", sep=';', dtype='float')

# Plot the data
fig, ax = plt.subplots()
groups = data.groupby('Play')
for name, group in groups:
    ax.plot(group.iloc[:, 0], group.iloc[:, 1], marker='o', linestyle='', label=name)
ax.set_xlabel('Temperature')
ax.set_ylabel('Humidity')
ax.legend()

# Split the data
X = data.iloc[:, 0:2]
y = data.iloc[:, 2]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=4)

# Train the model
model = DecisionTreeClassifier(criterion="gini", min_samples_split=20, min_samples_leaf=1)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)  
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Plot the tree
dot_data = export_graphviz(model, out_file=None, feature_names=X_train.columns.values, \
                           proportion=True, rounded=True)
graph = pydotplus.graph_from_dot_data(dot_data)  
# Visualize using IPython
#from IPython.display import Image, display
#display(Image(graph.create_png()))
# Visualize using matplotlib
from io import BytesIO
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
plt.figure()
img = mpimg.imread(BytesIO(graph.create_png()))
imgplot = plt.imshow(img,  aspect='equal')
plt.axis('off')
plt.show()

