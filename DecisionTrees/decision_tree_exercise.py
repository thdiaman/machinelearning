import pydotplus
import numpy as np
import pandas as pd
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split  
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import classification_report, confusion_matrix

# Read the data
iris = datasets.load_iris()
data = pd.DataFrame(data=np.c_[iris['data'], iris['target']], columns=iris['feature_names'] + ['target'])

# Plot the data
fig, ax = plt.subplots()
groups = data.groupby('target')
for name, group in groups:
    ax.plot(group.iloc[:, 0], group.iloc[:, 1], marker='o', linestyle='', label=name)
ax.set_xlabel('Sepal length')
ax.set_ylabel('Sepal width')
ax.legend(iris['target_names'])

# Split the data
X = data.iloc[:, 0:2]
y = data.iloc[:, 4]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Train the model
model = DecisionTreeClassifier(criterion="gini", min_samples_split=2, min_samples_leaf=1)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)  
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names = iris['target_names']))

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

