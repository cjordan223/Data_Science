# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Read the CPU performance data
df = pd.read_csv("https://raw.githubusercontent.com/grbruns/cst383/master/machine.csv")
df.index = df['vendor'] + ' ' + df['model']
df.drop(['vendor', 'model'], axis=1, inplace=True)

# Add the 'clock speed' feature
df['cs'] = np.round(1e3 / df['myct'], 2)  # clock speed in MHz (millions of cycles/sec)

# Display first few rows of the dataframe
df.head()

# Display information about the dataframe
df.info()

# Create a matrix of scatter plots using the Seaborn pairplot function
sns.pairplot(df)
plt.show()

# Pick a feature and produce a scatterplot with that feature on the x axis and ‘prp’ on the y axis
feature = 'mmin'  # You can change this to any other feature
sns.scatterplot(x=df[feature], y=df['prp'])
plt.xlabel(feature)
plt.ylabel('prp')
plt.title(f'Scatterplot of {feature} vs prp')
plt.show()

# Fit a linear model to the data
X = df[[feature]]
y = df['prp']
model = LinearRegression()
model.fit(X, y)
fit = model

# Plot the predicted relationship on top of your scatterplot
sns.scatterplot(x=df[feature], y=df['prp'])
plt.plot(df[feature], model.predict(X), color='red', linewidth=2)
plt.xlabel(feature)
plt.ylabel('prp')
plt.title(f'Scatterplot of {feature} vs prp with linear regression line')
plt.show()

# Compare with Seaborn's regplot
sns.regplot(x=df[feature], y=df['prp'])
plt.xlabel(feature)
plt.ylabel('prp')
plt.title(f'Seaborn regplot of {feature} vs prp')
plt.show()

# Create a scatterplot with the actual 'prp' value on the X axis and the predicted 'prp' value on the Y axis
predictions = model.predict(X)
sns.scatterplot(x=y, y=predictions)
plt.xlabel('Actual prp')
plt.ylabel('Predicted prp')
plt.title('Actual vs Predicted prp')
plt.show()

# Try with different features and multiple predictors
features = ['mmin', 'mmax', 'cach', 'chmin', 'chmax']  # Example list of features
X_multi = df[features]
model_multi = LinearRegression()
model_multi.fit(X_multi, y)
predictions_multi = model_multi.predict(X_multi)

# Scatterplot of actual vs predicted values with multiple predictors
sns.scatterplot(x=y, y=predictions_multi)
plt.xlabel('Actual prp')
plt.ylabel('Predicted prp')
plt.title('Actual vs Predicted prp with multiple predictors')
plt.show()

