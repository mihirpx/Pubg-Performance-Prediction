import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from sklearn.ensemble import GradientBoostingRegressor

Data=pd.read_csv('data.csv')
Data

Data.describe()

Data.info()

Data.isnull().sum()

Data.columns

Data = Data.drop(Data.columns[0], axis=1)
Data

Data.dropna(inplace=True) 

Data.isnull().sum()

# Distribution of winPoints
plt.figure(figsize=(6, 4))
sns.histplot(Data['winPoints'], kde=True, bins=30, color='Red')
plt.title('Distribution of winPoints')
plt.xlabel('winPoints')
plt.ylabel('Frequency')
plt.show()


#Scatterplot: Distance Traveled vs. Kills
plt.figure(figsize=(10, 6))
sns.scatterplot(x='walkDistance', y='kills', data=Data, alpha=0.7, color='green')
plt.title('Distance Traveled vs. Kills', fontsize=16)
plt.xlabel('Walk Distance (m)', fontsize=12)
plt.ylabel('Kills', fontsize=12)
plt.show()

#KDE Plot for Aggressive vs. Passive Players
plt.figure(figsize=(10, 6))
sns.kdeplot(Data[Data['kills'] > 0]['walkDistance'], label='Aggressive Players', shade=True, color='red')
sns.kdeplot(Data[Data['kills'] == 0]['walkDistance'], label='Passive Players', shade=True, color='blue')
plt.title('KDE Plot: Aggressive vs. Passive Players', fontsize=16)
plt.xlabel('Walk Distance (m)', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.legend()
plt.show()

#Kill Count by Match Type (Bar Chart)
kill_by_type = Data.groupby('matchType')['kills'].mean().sort_values()
plt.figure(figsize=(10, 6))
kill_by_type.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Average Kills by Match Type', fontsize=16)
plt.xlabel('Match Type', fontsize=12)
plt.ylabel('Average Kills', fontsize=12)
plt.xticks(rotation=45)
plt.show()

# Boxplot for winPoints by matchType
plt.figure(figsize=(12, 6))
sns.boxplot(x='matchType', y='winPoints', data=Data, palette='Set3')
plt.title('winPoints by Match Type')
plt.xlabel('Match Type')
plt.ylabel('winPoints')
plt.xticks(rotation=45)
plt.show()

#Label Encoding
label_encoder = preprocessing.LabelEncoder()
Data['matchType']= label_encoder.fit_transform(Data['matchType'])
Data['matchType'].unique()
Data

matchType='matchType'
print(Data[matchType])

Data.drop('Id',axis=1,inplace=True)
Data.drop('groupId',axis=1,inplace=True)
Data.drop('matchId',axis=1,inplace=True)
Data

# Correlation analysis
correlation_matrix = Data.corr()
# Correlation heatmap
plt.figure(figsize=(20, 15))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', vmin=-1, vmax=1)
plt.title('Correlation Heatmap')
plt.show()

# Correlation of winPoints with other features
winPoints_corr = correlation_matrix['winPoints'].sort_values(ascending=False)
print("\nCorrelation of winPoints with other features:")
print(winPoints_corr)

# Feature selection: Choose the features you want to use for prediction
features = ['kills', 'damageDealt', 'headshotKills', 'matchDuration', 'maxPlace', 'weaponsAcquired', 'killPlace','teamKills','revives','killStreaks']  
X = Data[features]
X

# Target variable: winPlacePerc
y = Data['winPlacePerc']
y
# Log transformation to stabilize variance and reduce skewness
# Handling skewed distribution: Apply log transformation to winPoints
y = np.log1p(y)
y

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizing the features (important for gradient boosting)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize the Gradient Boosting Regressor model
gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)

# Fit the model to the training data
gb_model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = gb_model.predict(X_test_scaled)
y_pred

# Evaluate the model performance
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error (MSE): {mse}")
rmse = mse ** 0.5
print(f"Root Mean Squared Error (RMSE): {rmse}")
r2 = r2_score(y_test, y_pred)
print(f"R-squared (R²): {r2}")

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
import pandas as pd

# Assuming the model is already trained
# And X_test_scaled and y_test_exp are your test data

# Predict the results using the trained model
y_pred_exp = gb_model.predict(X_test_scaled)

# Calculating the metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print out the metrics
print("Model Evaluation:")
print("-------------------")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"R-squared (R2): {r2:.4f}")

# Feature Importance (for tree-based models like Gradient Boosting)
feature_importances = gb_model.feature_importances_

# Create a DataFrame for feature importance
feature_importance_df = pd.DataFrame({
    'Feature': X.columns,  # Use the feature names from X (training data)
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

# Display the top features based on importance
print("\nFeature Importance:")
print(feature_importance_df)

# Display model parameters
print("\nModel Parameters:")
print(gb_model.get_params())

# Plot Actual vs Predicted
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([0, max(y_test)], [0, max(y_test)], color='red', linestyle='--')
plt.title('Actual vs Predicted winPlacePerc')
plt.xlabel('Actual winPlacePerc')
plt.ylabel('Predicted winPlacePerc')
plt.show()

# Residual plot
residuals = y_test - y_pred
print(residuals)
plt.figure(figsize=(8, 6))
plt.scatter(y_pred, residuals, alpha=0.5)
plt.hlines(y=0, xmin=min(y_pred), xmax=max(y_pred), colors='red', linestyles='--')
plt.title('Residuals Plot')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.show()

# Cross-validation to check stability of the model
from sklearn.model_selection import cross_val_score
cv_scores = cross_val_score(gb_model, X_train_scaled, y_train, cv=5, scoring='neg_mean_squared_error')
print(f"Cross-validated Mean Squared Error: {-cv_scores.mean()}")

#Check multicollinearity
X_with_const = add_constant(X_train)  # Adding constant to the feature set
vif = pd.DataFrame()
vif["Features"] = X_with_const.columns
vif["VIF"] = [variance_inflation_factor(X_with_const.values, i) for i in range(X_with_const.shape[1])]
print(vif)

# Plot feature importance
importances = gb_model.feature_importances_
plt.figure(figsize=(10, 6))
sns.barplot(x=importances, y=X.columns)
plt.title('Feature Importance')
plt.show()

# Residuals
residuals = y_test - y_pred

# Plot residuals
plt.figure(figsize=(8, 6))
sns.histplot(residuals, kde=True)
plt.title('Residual Distribution')
plt.xlabel('Residuals')
plt.show()

import joblib
joblib.dump(gb_model, 'gradient_boosting_model.pkl')

  
