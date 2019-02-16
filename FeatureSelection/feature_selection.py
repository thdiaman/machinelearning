import numpy as np
import pandas as pd
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split  

# Read the data
iris = datasets.load_iris()
data = pd.DataFrame(data=np.c_[iris['data'], iris['target']], columns=iris['feature_names'] + ['target'])
X, y = data.iloc[:, 0:4], data.iloc[:, 4]

# Print the correlation matrix
pd.set_option('display.width', 100)
print(X.corr())

# Plot the data
axes = pd.plotting.scatter_matrix(X)
plt.show()
