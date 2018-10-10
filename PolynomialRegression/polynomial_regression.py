import numpy as np
import pandas as pd
from math import sqrt
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Read data
data = pd.read_csv("ac_data.csv", sep=';')
X = data.iloc[:, 0].values.reshape(-1, 1)
y = data.iloc[:, 1]

# Create test space
X_test = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)

# Apply linear regression model
model = LinearRegression()
model.fit(X, y)
y_pred_d1 = model.predict(X_test)

# Show equation for first model
print('Equation: y = ' + str(model.intercept_) + ' + ' + str(model.coef_))

# Apply polynomial regression model
degree = 4
poly_features = PolynomialFeatures(degree = degree, include_bias = False)
model = LinearRegression()
model.fit(poly_features.fit_transform(X), y)
y_pred_d3 = model.predict(poly_features.fit_transform(X_test))

# Show equation for second model
print('Equation: y = ' + str(model.intercept_) + ' + ' + \
        ' + '.join(str(model.coef_[d]) + ' * x^' + str(d+1) for d in range(degree)))

# Plot the two models
plt.plot(X, y, 'o')
plt.plot(X_test, y_pred_d1, linewidth=3, label="degree=1")
plt.plot(X_test, y_pred_d3, linewidth=3, label="degree=" + str(degree))
plt.legend()
plt.xlabel(data.columns.values[0])
plt.ylabel(data.columns.values[1])
plt.show()

# Calculate RMSE for different polynomial degrees
errors = []
degrees = list(range(1, 11))
for degree in degrees:
	poly_features = PolynomialFeatures(degree = degree)
	X_poly = poly_features.fit_transform(X)
	model = LinearRegression()
	model.fit(X_poly, y)
	y_pred = model.predict(X_poly)
	errors.append(sqrt(mean_squared_error(y, y_pred)))

plt.plot(degrees, errors)
plt.xlabel('Degree')
plt.ylabel('RMSE')
plt.show()
