import pydotplus
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_graphviz

# Read the data
data = pd.read_csv("weather.csv", sep=';', dtype='category')

# Transform the data
#print(data)
data = data.apply(lambda x: pd.factorize(x)[0])

# Split the data
X_train = data.iloc[:, 0:3]
Y_train = data[["Play"]]

# Train the classifier
model = DecisionTreeClassifier(criterion="gini", min_samples_split=2, min_samples_leaf=1)
model.fit(X_train, Y_train)

# Plot the tree
dot_data = export_graphviz(model, out_file=None, feature_names=X_train.columns.values, \
                           class_names=["No", "Yes"], proportion=True, rounded=True)
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

# Make a new prediction
#print(model.predict([[0, 0, 0]]))
