# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Read the CPU data
df = pd.read_csv("https://raw.githubusercontent.com/grbruns/cst383/master/machine.csv")
df.index = df['vendor'] + ' ' + df['model']
df.drop(['vendor', 'model'], axis=1, inplace=True)
df['cs'] = np.round(1e3 / df['myct'], 2)  # clock speed in MHz

# Create training and test sets
X = df.drop(columns='prp').values
y = df['prp'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define a function to calculate RMSE
def calculate_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# Experiment with building linear models using interactions and nonlinear transformations

# Experiment 1: Simple linear model with original features
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
rmse1 = calculate_rmse(y_test, y_pred)
print("Experiment 1 - RMSE:", rmse1)

# Experiment 2: Adding interaction term between 'mmin' and 'mmax'
df['interaction'] = df['mmin'] * df['mmax']
X = df.drop(columns='prp').values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
rmse2 = calculate_rmse(y_test, y_pred)
print("Experiment 2 - RMSE with interaction term:", rmse2)

# Experiment 3: Nonlinear transformation (log) on 'mmin' and 'mmax'
df['log_mmin'] = np.log(df['mmin'])
df['log_mmax'] = np.log(df['mmax'])
X = df.drop(columns='prp').values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
rmse3 = calculate_rmse(y_test, y_pred)
print("Experiment 3 - RMSE with log transformation:", rmse3)

# Experiment 4: Combining interaction and nonlinear transformations
df['interaction_log'] = np.log(df['mmin']) * np.log(df['mmax'])
X = df.drop(columns='prp').values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
rmse4 = calculate_rmse(y_test, y_pred)
print("Experiment 4 - RMSE with interaction and log transformation:", rmse4)

# Experiment 5: Polynomial transformation (squared term) on 'mmin'
df['mmin_squared'] = df['mmin'] ** 2
X = df.drop(columns='prp').values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
rmse5 = calculate_rmse(y_test, y_pred)
print("Experiment 5 - RMSE with polynomial transformation:", rmse5)

# Identify the best RMSE score
experiments = {
    "Simple linear model": rmse1,
    "Interaction term": rmse2,
    "Log transformation": rmse3,
    "Interaction + Log transformation": rmse4,
    "Polynomial transformation": rmse5
}

best_experiment = min(experiments, key=experiments.get)
best_rmse = experiments[best_experiment]

print(f"Best experiment: {best_experiment} with RMSE: {best_rmse}")

