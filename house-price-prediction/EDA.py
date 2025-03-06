# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib

# Load the dataset
data = pd.read_csv('real_estate_dataset.csv')

# Outlier Detection (after filling missing values)
Q1 = data.quantile(0.25)
Q3 = data.quantile(0.75)
IQR = Q3 - Q1
outliers = ((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).sum()
print(f"Outliers detected in each feature:\n{outliers}")

# Data Exploration and Preprocessing
data.drop(['ID'], axis=1, inplace=True)  # Drop 'ID' column
data.fillna(data.median(), inplace=True)  # Handle missing values

# Check for NaN values in the features and target variable
print("Missing values in features:")
print(data.isna().sum())  # Check for NaN in the features

# Separate features (X) and target variable (y)
X = data.drop(['Price'], axis=1)
y = data['Price']

# Check for NaN values in the target variable
print("\nMissing values in target (Price):")
print(y.isna().sum())  # Check for NaN in the target

# Encoding categorical features (if applicable)
X = pd.get_dummies(X, drop_first=True)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling (Optional)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaler
joblib.dump(scaler, 'scaler.pkl')

# Model: Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)
y_pred_lr = lr_model.predict(X_test_scaled)


# Evaluate Linear Regression model
def evaluate_model(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    print(f"\n{model_name} Performance Metrics:")
    print(f"MAE: {mae:.2f}, MSE: {mse:.2f}, RMSE: {rmse:.2f}, R²: {r2:.2f}")
    return mae, mse, rmse, r2


evaluate_model(y_test, y_pred_lr, "Linear Regression")

# Model: Random Forest (Before Hyperparameter Tuning)
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# Evaluate Random Forest model (Before Tuning)
evaluate_model(y_test, y_pred_rf, "Random Forest (Before Tuning)")

# Hyperparameter Tuning for Random Forest (Grid Search)
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=5, n_jobs=-1,
                           verbose=2, error_score='raise')

grid_search.fit(X_train, y_train)

# Best Random Forest model from Grid Search
best_rf_model = grid_search.best_estimator_
print(f"Best Hyperparameters: {grid_search.best_params_}")

y_pred_best_rf = best_rf_model.predict(X_test)
evaluate_model(y_test, y_pred_best_rf, "Random Forest (After Tuning)")

# Model Comparison: SVR
svr_model = SVR()
svr_model.fit(X_train_scaled, y_train)
y_pred_svr = svr_model.predict(X_test_scaled)
evaluate_model(y_test, y_pred_svr, "SVR")

# Evaluation Metrics Visualization
metrics_df = pd.DataFrame({
    'Model': ['Linear Regression', 'Random Forest (Before Tuning)', 'Random Forest (After Tuning)', 'SVR'],
    'MAE': [mean_absolute_error(y_test, y_pred_lr), mean_absolute_error(y_test, y_pred_rf),
            mean_absolute_error(y_test, y_pred_best_rf), mean_absolute_error(y_test, y_pred_svr)],
    'MSE': [mean_squared_error(y_test, y_pred_lr), mean_squared_error(y_test, y_pred_rf),
            mean_squared_error(y_test, y_pred_best_rf), mean_squared_error(y_test, y_pred_svr)],
    'RMSE': [np.sqrt(mean_squared_error(y_test, y_pred_lr)), np.sqrt(mean_squared_error(y_test, y_pred_rf)),
             np.sqrt(mean_squared_error(y_test, y_pred_best_rf)), np.sqrt(mean_squared_error(y_test, y_pred_svr))],
    'R²': [r2_score(y_test, y_pred_lr), r2_score(y_test, y_pred_rf), r2_score(y_test, y_pred_best_rf),
           r2_score(y_test, y_pred_svr)]
})

metrics_df.set_index('Model').plot(kind='bar', figsize=(10, 6))
plt.title("Model Comparison: MAE, MSE, RMSE, and R²")
plt.ylabel("Score")
plt.show()

# Residual Plot for Linear Regression
residuals_lr = y_test - y_pred_lr
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_pred_lr, y=residuals_lr)
plt.axhline(y=0, color='black', linestyle='--')
plt.title("Linear Regression: Residual Plot")
plt.xlabel("Predicted Prices")
plt.ylabel("Residuals")
plt.show()

# Feature Importance from Random Forest (After Tuning)
feature_importance = best_rf_model.feature_importances_
plt.figure(figsize=(10, 6))
plt.barh(X.columns, feature_importance)
plt.title("Feature Importance from Random Forest")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.show()

# Save the tuned Random Forest model
joblib.dump(best_rf_model, 'random_forest_model.pkl')
