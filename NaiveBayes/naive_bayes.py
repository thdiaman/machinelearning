import pandas as pd
from sklearn.naive_bayes import BernoulliNB

# Read the data
data = pd.read_csv("traffic.csv", sep=';', dtype='category')

# Transform the data
data = data.apply(lambda x: pd.factorize(x)[0])

# Split the data
X = data.iloc[:, 0:2]
Y = data.iloc[:, 2]

# Train the model (set also alpha for smoothing)
model = BernoulliNB()
model.fit(X, Y)

# Make predictions
print(model.predict([[0, 0]]))
print(model.predict_proba([[0, 0]]))

