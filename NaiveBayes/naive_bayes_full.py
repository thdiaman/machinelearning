import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split  
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, precision_recall_curve, auc

# Read the data
data = pd.read_csv("trafficfull.csv", sep=';')
X = data.iloc[:, 0:-1]
y = data.iloc[:, -1]

# Plot the data
fig, ax = plt.subplots()
groups = data.groupby('HighTraffic')
for name, group in groups:
    ax.plot(group.iloc[:, 0], group.iloc[:, 1], marker='o', linestyle='', label=name)
ax.set_xlabel('Temperature')
ax.set_ylabel('Wind')
ax.legend()

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Train the model
model = GaussianNB()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)  
y_pred_prob = model.predict_proba(X_test)  
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Plot the Precision Recall Curve
#precision, recall, thresholds = precision_recall_curve(y_test, y_pred_prob[:, 1])
#precision = np.insert(precision, 0, 0)
#recall = np.insert(recall, 0, 1)
#plt.plot(precision, recall, 'b')
#plt.ylabel('Precision')
#plt.xlabel('Recall')
#plt.show()

# Plot the ROC curve
fpr, tpr, threshold = roc_curve(y_test, y_pred_prob[:, 1])
plt.figure()
plt.plot(fpr, tpr, 'b')
plt.plot([0, 1], [0, 1],'r--')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
print(auc(fpr, tpr)) # print the AUC
