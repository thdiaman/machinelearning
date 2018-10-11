import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split  
from sklearn.feature_selection import SelectKBest, chi2

# Read the data
iris = datasets.load_iris()
data = pd.DataFrame(data=np.c_[iris['data'], iris['target']], \
                    columns=iris['feature_names'] + ['target'])
# Split the data
trainingdata,testdata = train_test_split(data,test_size=0.30,random_state=2)
X_train, y_train = trainingdata.iloc[:, 0:4], trainingdata.iloc[:, 4]
X_test, y_test = testdata.iloc[:, 0:4], testdata.iloc[:, 4]

# Apply classifier
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)  
print(classification_report(y_test, y_pred))

# Keep 3 out of 4 features
fselector = SelectKBest(chi2, k=3)
fselector.fit(X_train, y_train)
# and apply classifier
model = DecisionTreeClassifier()
model.fit(fselector.transform(X_train), y_train)
y_pred = model.predict(fselector.transform(X_test))
print(classification_report(y_test, y_pred))

# Keep 2 out of 4 features
fselector = SelectKBest(chi2, k=2)
fselector.fit(X_train, y_train)
# and apply classifier
model = DecisionTreeClassifier()
model.fit(fselector.transform(X_train), y_train)
y_pred = model.predict(fselector.transform(X_test))
print(classification_report(y_test, y_pred))

# Keep 1 out of 4 features
fselector = SelectKBest(chi2, k=2)
fselector.fit(X_train, y_train)
# and apply classifier
model = DecisionTreeClassifier()
model.fit(fselector.transform(X_train), y_train)
y_pred = model.predict(fselector.transform(X_test))
print(classification_report(y_test, y_pred))

