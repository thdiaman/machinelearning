import numpy as np
import pandas as pd
from math import sqrt
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from mpl_toolkits.mplot3d import Axes3D

# Read data
data = pd.read_csv("data3d.csv", sep=';')
X = data.iloc[:, 0:2]
y = data.iloc[:, 2]

# Plot data
#fig = plt.figure()
#ax = Axes3D(fig)
#ax.scatter(x, y, z, color='b') 
#plt.show()

# Apply linear regression model
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

# Show equation for first model
print('Equation: y = ' + str(model.intercept_) + ' + ' + \
        str(model.coef_[0]) + ' * x ' + ' + ' + \
        str(model.coef_[1]) + ' * y ')

# Calculate metrics
print("Mean Absolute Error: %.2f" %mean_absolute_error(y, y_pred))
print("Mean Squared Error: %.2f" %mean_squared_error(y, y_pred))
print("Root Mean Squared Error: %.2f" %sqrt(mean_squared_error(y, y_pred)))
print("Coefficient of Determination: %.2f" %r2_score(y, y_pred))

# Apply polynomial regression model
degree = 2
poly_features = PolynomialFeatures(degree = degree, include_bias = False)
model = LinearRegression()
model.fit(poly_features.fit_transform(X), y)
y_pred = model.predict(poly_features.fit_transform(X))

# Show equation for second model
print('\nEquation: y = ' + str(model.intercept_) + ' + ' + \
        str(model.coef_[0]) + ' * x ' + ' + ' + \
        str(model.coef_[1]) + ' * y ' + ' + ' + \
        str(model.coef_[2]) + ' * x^2 ' + ' + ' + \
        str(model.coef_[3]) + ' * xy ' + ' + ' + \
        str(model.coef_[4]) + ' * y^2 ')

# Calculate metrics
print("Mean Absolute Error: %.2f" %mean_absolute_error(y, y_pred))
print("Mean Squared Error: %.2f" %mean_squared_error(y, y_pred))
print("Root Mean Squared Error: %.2f" %sqrt(mean_squared_error(y, y_pred)))
print("Coefficient of Determination: %.2f" %r2_score(y, y_pred))
