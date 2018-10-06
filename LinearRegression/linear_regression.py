import pandas as pd
from math import sqrt
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split  
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Read data
data = pd.read_csv("houses.csv", sep=';')
X = data.iloc[:, 0].values.reshape(-1, 1)
y = data.iloc[:, 1]

# Fit model to data
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

# Plot data and model
fig, ax = plt.subplots()
ax.scatter(X, y, color='black')
ax.plot(X, y_pred, color='blue', linewidth=3)
ax.set_xlabel(data.columns.values[0])
ax.set_ylabel(data.columns.values[1])
#plt.show()

# Show equation
print('Equation: y = %.2f + %.2f * x' %(model.intercept_, model.coef_))

# Calculate metrics
print("Mean Absolute Error: %.2f" %mean_absolute_error(y, y_pred))
print("Mean Squared Error: %.2f" %mean_squared_error(y, y_pred))
print("Root Mean Squared Error: %.2f" %sqrt(mean_squared_error(y, y_pred)))
print("Coefficient of Determination: %.2f" %r2_score(y, y_pred))
