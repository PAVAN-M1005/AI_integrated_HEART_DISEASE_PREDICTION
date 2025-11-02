# train_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib
import os
import sys

# --- 1. Load Dataset ---
DATA_FILE = "heart.csv"

if not os.path.exists(DATA_FILE):
    print(f"❌ Error: '{DATA_FILE}' not found in the current folder.")
    sys.exit(1)

print("📥 Loading dataset...")
df = pd.read_csv(DATA_FILE)

# --- 2. Normalize Column Names ---
df.columns = df.columns.str.strip().str.lower()
print("📋 Columns found:", df.columns.tolist())

# --- 3. Define Features and Target (based on your dataset) ---
# NOTE: These features are based on the inputs in your index.html form.
FEATURES = [
    'age', 'sex', 'chestpaintype', 'restingbp', 'cholesterol',
    'fastingbs', 'restingecg', 'maxhr', 'exerciseangina',
    'oldpeak', 'st_slope'
]
TARGET = 'heartdisease'

# --- 4. Check Required Columns ---
missing = [col for col in FEATURES + [TARGET] if col not in df.columns]
if missing:
    print(f"❌ Error: Missing required columns: {missing}")
    sys.exit(1)

# --- 5. Split Features and Target ---
X = df[FEATURES]
y = df[TARGET]

# --- 6. Encode Categorical Columns ---
# Convert categorical (string) columns into numeric dummies
# This is crucial for the Logistic Regression model.
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
X = pd.get_dummies(X, columns=categorical_cols, drop_first=False) # Keep all dummies for consistent feature ordering

# --- 7. Split Data ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"📊 Dataset split into {len(X_train)} training and {len(X_test)} testing samples.")

# --- 8. Scale Numerical Columns ---
# Identify numeric columns for scaling (excluding dummy variables)
# We only want to scale the original numeric features: age, restingbp, cholesterol, maxhr, oldpeak
numeric_cols_to_scale = ['age', 'restingbp', 'cholesterol', 'maxhr', 'oldpeak']
scaler = StandardScaler()

# Ensure only the intended columns are scaled
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

X_train_scaled[numeric_cols_to_scale] = scaler.fit_transform(X_train[numeric_cols_to_scale])
X_test_scaled[numeric_cols_to_scale] = scaler.transform(X_test[numeric_cols_to_scale])

# --- 9. Train Model ---
print("🚀 Training Logistic Regression model...")
model = LogisticRegression(max_iter=1000, solver='liblinear', random_state=42) # Added solver and random_state for consistency
model.fit(X_train_scaled, y_train)
print("✅ Model training complete.")

# --- 10. Evaluate Model ---
train_acc = model.score(X_train_scaled, y_train)
test_acc = model.score(X_test_scaled, y_test)
print(f"📈 Training Accuracy: {train_acc * 100:.2f} %")
print(f"📊 Testing Accuracy: {test_acc * 100:.2f} %")

# --- 11. Save Model and Scaler ---
MODEL_FILE = "heart_disease_model.pkl"
SCALER_FILE = "heart_disease_scaler.pkl"
FEATURES_FILE = "model_features.pkl"

joblib.dump(model, MODEL_FILE)
joblib.dump(scaler, SCALER_FILE)
# Save the final list of columns/features after one-hot encoding for use in app.py
joblib.dump(X_train_scaled.columns.tolist(), FEATURES_FILE)

print(f"\n🎉 Success! ML assets saved:")
print(f" - Model: {MODEL_FILE}")
print(f" - Scaler: {SCALER_FILE}")
print(f" - Features List: {FEATURES_FILE}")

print("\nNEXT STEP: You can now run the Flask application with 'python app.py'.")
