# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Read the CPU data
df = pd.read_csv("https://raw.githubusercontent.com/grbruns/cst383/master/machine.csv")
df.index = df['vendor'] + ' ' + df['model']
df.drop(['vendor', 'model'], axis=1, inplace=True)
df['cs'] = np.round(1e3 / df['myct'], 2)  # clock speed in MHz

# Define a function to calculate RMSE manually
def calculate_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# Split the data randomly into a training set and a test set, using a 70/30 split
def split_and_train(df, predictors):
    X = df[predictors].values
    y = df['prp'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=None)
    
    # Create a linear model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Compute the RMSE manually on the test data
    y_pred = model.predict(X_test)
    rmse = calculate_rmse(y_test, y_pred)
    
    return rmse

# Step 1: Initial split and training
predictors = ['mmin', 'mmax']  # Choose predictor variables
initial_rmse = split_and_train(df, predictors)
print("Initial RMSE:", initial_rmse)

# Step 2: Repeat steps 2-4 with new randomly-generated test and training sets
new_rmse = split_and_train(df, predictors)
print("New RMSE with different split:", new_rmse)

# Step 3: Do steps 2-4 100 times, each time creating different training/test sets
rmse_values = []
for _ in range(100):
    rmse_values.append(split_and_train(df, predictors))

# Plot RMSE values on a histogram
plt.hist(rmse_values, bins=20, edgecolor='k')
plt.xlabel('RMSE')
plt.ylabel('Frequency')
plt.title('Distribution of RMSE values (100 trials, 70/30 split)')
plt.show()

# Step 4: Repeat with an 80/20 split
def split_and_train_80_20(df, predictors):
    X = df[predictors].values
    y = df['prp'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None)
    
    # Create a linear model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Compute the RMSE manually on the test data
    y_pred = model.predict(X_test)
    rmse = calculate_rmse(y_test, y_pred)
    
    return rmse

# Do steps 2-4 100 times with 80/20 split
rmse_values_80_20 = []
for _ in range(100):
    rmse_values_80_20.append(split_and_train_80_20(df, predictors))

# Plot RMSE values on a histogram for 80/20 split
plt.hist(rmse_values_80_20, bins=20, edgecolor='k')
plt.xlabel('RMSE')
plt.ylabel('Frequency')
plt.title('Distribution of RMSE values (100 trials, 80/20 split)')
plt.show()

# Step 5: Compute MSE using cross-validation on the entire data set
X = df[predictors].values
y = df['prp'].values
model = LinearRegression()
mse_scores = -cross_val_score(model, X, y, cv=10, scoring='neg_mean_squared_error')
rmse_scores = np.sqrt(mse_scores)

# Plot all the histogram values using a histogram
plt.hist(rmse_scores, bins=20, edgecolor='k')
plt.xlabel('RMSE')
plt.ylabel('Frequency')
plt.title('Distribution of RMSE values using cross-validation')
plt.show()

