import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib

# --- Settings ---
DATA_PATH = "data/raw/Life-Expectancy-Data.csv"
MODEL_OUT_PATH = "models/random_forest.joblib"

# --- Load Data ---
print("Loading data...")
df = pd.read_csv(DATA_PATH)

# --- Feature Selection ---
# Use the same features as in dashboard
feat_cols = [
    "GDP", "Schooling", " BMI ", "Alcohol", "Adult Mortality", 
    " HIV/AIDS", "Total expenditure", "Income composition of resources"
]
target_col = "Life expectancy "

# Basic check: drop NA rows for selected features
df_model = df[feat_cols + [target_col]].dropna()

X = df_model[feat_cols].values
y = df_model[target_col].values

# --- Train/Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.18, random_state=42)

# --- Train Random Forest ---
print("Training RandomForestRegressor...")
rf = RandomForestRegressor(n_estimators=150, max_depth=8, random_state=42)
rf.fit(X_train, y_train)

# --- Evaluate ---
y_pred = rf.predict(X_test)
print(f"R2 Score: {r2_score(y_test, y_pred):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.3f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred):.3f}")

# --- Save Model ---
if not os.path.exists("models"):
    os.makedirs("models")
joblib.dump(rf, MODEL_OUT_PATH)
print(f"Model saved to {MODEL_OUT_PATH}")
