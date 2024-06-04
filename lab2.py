# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Read the CPU data
df = pd.read_csv("https://raw.githubusercontent.com/grbruns/cst383/master/machine.csv")
df.index = df['vendor'] + ' ' + df['model']
df.drop(['vendor', 'model'], axis=1, inplace=True)
df['cs'] = np.round(1e3 / df['myct'], 2)  # clock speed in MHz

# Create NumPy arrays X and y from the data
y = df['prp'].values
X = df[['mmin', 'mmax']].values  # Choose two columns, you can change these

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create a linear model
model = LinearRegression()
model.fit(X_train, y_train)

# Get the coefficients of the model
coefficients = model.coef_
intercept = model.intercept_
print("Coefficients:", coefficients)
print("Intercept:", intercept)

# Check the importance of predictor variables
print("Importance of predictors: Both coefficients matter in predicting 'prp'.")

# Calculate the R-squared value
r_squared = model.score(X_train, y_train)
print("R-squared value:", r_squared)
print("Did you get a good R-squared value? The best possible R-squared value is 1.")

# Produce a scatterplot of predicted vs actual prp values
y_pred = model.predict(X_test)
sns.scatterplot(x=y_pred, y=y_test)
plt.xlabel('Predicted prp')
plt.ylabel('Actual prp')
plt.title('Predicted vs Actual prp')
plt.show()

# Repeat with a different pair of predictor variables
X = df[['cach', 'chmin']].values  # Choose another pair of columns
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model.fit(X_train, y_train)
coefficients = model.coef_
intercept = model.intercept_
print("Coefficients (different predictors):", coefficients)
print("Intercept (different predictors):", intercept)
r_squared = model.score(X_train, y_train)
print("R-squared value (different predictors):", r_squared)
y_pred = model.predict(X_test)
sns.scatterplot(x=y_pred, y=y_test)
plt.xlabel('Predicted prp')
plt.ylabel('Actual prp')
plt.title('Predicted vs Actual prp (different predictors)')
plt.show()

# Repeat with all predictor variables
X = df.drop(columns='prp').values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model.fit(X_train, y_train)
coefficients = model.coef_
intercept = model.intercept_
print("Coefficients (all predictors):", coefficients)
print("Intercept (all predictors):", intercept)
r_squared = model.score(X_train, y_train)
print("R-squared value (all predictors):", r_squared)
y_pred = model.predict(X_test)
sns.scatterplot(x=y_pred, y=y_test)
plt.xlabel('Predicted prp')
plt.ylabel('Actual prp')
plt.title('Predicted vs Actual prp (all predictors)')
plt.show()

# Identify most important predictors
important_predictors = pd.Series(coefficients, index=df.drop(columns='prp').columns).sort_values(ascending=False)
print("Most important predictors:")
print(important_predictors)

