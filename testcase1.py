import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

# Load the data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Combine train and test datasets to perform preprocessing on both
all_data = pd.concat([train.drop('SalePrice', axis=1), test])

# Drop irrelevant columns
cols_to_drop = ['Id', 'Utilities', 'Street', 'PoolQC']
all_data.drop(cols_to_drop, axis=1, inplace=True)

# Fill missing values
all_data.fillna(value=0, inplace=True)

# Create new features
all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']
all_data['TotalBathrooms'] = all_data['FullBath'] + 0.5 * all_data['HalfBath'] + all_data['BsmtFullBath'] + 0.5 * all_data['BsmtHalfBath']
all_data['TotalPorchSF'] = all_data['OpenPorchSF'] + all_data['EnclosedPorch'] + all_data['3SsnPorch'] + all_data['ScreenPorch']

# One-hot encode categorical columns
all_data = pd.get_dummies(all_data)

# Scale numerical features
scaler = StandardScaler()
numerical_cols = all_data.select_dtypes(include=['int64', 'float64']).columns
all_data[numerical_cols] = scaler.fit_transform(all_data[numerical_cols])

# Reduce dimensionality using PCA
pca = PCA(n_components=50)
all_data = pca.fit_transform(all_data)

# Split data back into train and test sets
X_train = all_data[:train.shape[0], :]
X_test = all_data[train.shape[0]:, :]
y_train = train['SalePrice']

# Train the model
model = Ridge(alpha=10)
model.fit(X_train, y_train)

# Predict on the test set
y_test_pred_ridge = model.predict(X_test)

# Train Random Forest model with the best hyperparameters
param_grid = {
    'n_estimators': [100, 200, 500],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
rf_model = RandomForestRegressor(random_state=42)
rf_grid_search = GridSearchCV(rf_model, param_grid, cv=5, n_jobs=-1)
rf_grid_search.fit(X_train, y_train)

# Predict on the test set
rf_model = RandomForestRegressor(**rf_grid_search.best_params_, random_state=42)
rf_model.fit(X_train, y_train)
y_test_pred_rf = rf_model.predict(X_test)

# Combine the predictions
y_test_pred = 0.5 * y_test_pred_ridge + 0.5 * y_test_pred_rf

# Create a submission DataFrame
submission = pd.DataFrame({'Id': test['Id'], 'SalePrice': y_test_pred})

# Save the submission to a CSV file
submission.to_csv('submission.csv', index=False)
