import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import xgboost as xgb
import joblib

# Load dataset
df = pd.read_csv("real_dataset.csv")

# Select only numeric columns (ignore image)
X = df[['latitude', 'longitude', 'area_sqft', 'bedrooms', 'bathrooms']]
y = df['price']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale numerical data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train XGBoost model
model = xgb.XGBRegressor(
    n_estimators=600,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.9,
    colsample_bytree=0.9,
    objective="reg:squarederror"
)

print("Training numeric model...")
model.fit(X_train_scaled, y_train)

# Evaluate
pred = model.predict(X_test_scaled)
mae = mean_absolute_error(y_test, pred)
print("MAE:", mae)

# Save model + scaler
joblib.dump(model, "numeric_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Training completed — numeric_model.pkl saved!")
