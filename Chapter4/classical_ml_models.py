import pandas as pd

# Load training data
df1 = pd.read_csv("peptide_train_dataset.csv")

# Load testing data
df2 = pd.read_csv("peptide_test_dataset.csv")

# Concatenate the dataframes
df = pd.concat([df1, df2], ignore_index=True)
labels = df.pop('RT')


from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import xgboost as xgb
import pylab as pl
import numpy as np

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import xgboost as xgb
import numpy as np

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(df, labels, test_size=0.2, random_state=RSEED)

# 1. Random Forest with RandomizedSearchCV
random_grid = {
    'n_estimators': [int(x) for x in np.linspace(start=100, stop=1200, num=12)],
    'max_features': ['auto', 'sqrt'],
    'max_depth': [int(x) for x in np.linspace(5, 30, num=6)],
    'min_samples_split': [2, 5, 10, 15, 100],
    'min_samples_leaf': [1, 2, 5, 10]
}
rf = RandomForestRegressor(random_state=RSEED)
rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=100, cv=10, verbose=2, random_state=RSEED, n_jobs=1)
rf_random.fit(X_train, y_train)
print(f"Best hyperparameters for RF: {rf_random.best_params_}")

# 2. XGBoost with GridSearchCV
xgb_params = {
    'objective': ['reg:squarederror'],
    'colsample_bytree': [0.3, 0.5, 0.7],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'alpha': [1, 5, 10],
    'n_estimators': [10, 50, 100]
}
xgb_reg = xgb.XGBRegressor()
xgb_search = GridSearchCV(xgb_reg, xgb_params, cv=10, verbose=2, n_jobs=1)
xgb_search.fit(X_train, y_train)
print(f"Best hyperparameters for XGBoost: {xgb_search.best_params_}")

# 3. SVR with GridSearchCV
svr_params = {
    'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001],
    'kernel': ['rbf']
}
svr = SVR()
svr_search = GridSearchCV(svr, svr_params, cv=10, verbose=2, n_jobs=1)
svr_search.fit(X_train, y_train)
print(f"Best hyperparameters for SVR: {svr_search.best_params_}")

#Evaluate the model

import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

# 1. Random Forest Evaluation
rf_best = rf_random.best_estimator_
rf_preds = rf_best.predict(X_test)

rf_mse = mean_squared_error(y_test, rf_preds)
rf_r2 = r2_score(y_test, rf_preds)
print(f"Random Forest MSE: {rf_mse:.2f}")
print(f"Random Forest R2: {rf_r2:.2f}")

# 2. XGBoost Evaluation
xgb_best = xgb_search.best_estimator_
xgb_preds = xgb_best.predict(X_test)

xgb_mse = mean_squared_error(y_test, xgb_preds)
xgb_r2 = r2_score(y_test, xgb_preds)
print(f"XGBoost MSE: {xgb_mse:.2f}")
print(f"XGBoost R2: {xgb_r2:.2f}")

# 3. SVR Evaluation
svr_best = svr_search.best_estimator_
svr_preds = svr_best.predict(X_test)

svr_mse = mean_squared_error(y_test, svr_preds)
svr_r2 = r2_score(y_test, svr_preds)
print(f"SVR MSE: {svr_mse:.2f}")
print(f"SVR R2: {svr_r2:.2f}")

# Plotting Predicted vs. Actual values for each model
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
axes = axes.ravel()

# Plotting function
def plot_results(actual, predicted, model_name, ax):
    ax.scatter(actual, predicted, alpha=0.5)
    ax.plot([min(actual), max(actual)], [min(actual), max(actual)], color='red', linestyle='--')
    ax.set_xlabel('Actual Values')
    ax.set_ylabel('Predicted Values')
    ax.set_title(f"{model_name} - Actual vs. Predicted")
    
plot_results(y_test, rf_preds, "Random Forest", axes[0])
plot_results(y_test, xgb_preds, "XGBoost", axes[1])
plot_results(y_test, svr_preds, "SVR", axes[2])

plt.tight_layout()
plt.show()

# plots of performance

import matplotlib.pyplot as plt

# Visualization for Random Forest
plt.figure(figsize=(10, 6))
plt.plot(X_test, y_test, 'ro', label='Actual values')
plt.plot(X_test, rf_preds, 'bo', label='Predicted values')
plt.xlabel('Data')
plt.ylabel('Target')
plt.title('Random Forest Regression')
plt.legend()
plt.show()

# Visualization for XGBoost
plt.figure(figsize=(10, 6))
plt.plot(X_test, y_test, 'ro', label='Actual values')
plt.plot(X_test, xgb_preds, 'bo', label='Predicted values')
plt.xlabel('Data')
plt.ylabel('Target')
plt.title('XGBoost Regression')
plt.legend()
plt.show()

# Visualization for SVM
plt.figure(figsize=(10, 6))
plt.plot(X_test, y_test, 'ro', label='Actual values')
plt.plot(X_test, svm_preds, 'bo', label='Predicted values')
plt.xlabel('Data')
plt.ylabel('Target')
plt.title('SVM Regression')
plt.legend()
plt.show()
