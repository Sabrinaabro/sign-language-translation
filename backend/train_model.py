import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# Load dataset (no header, so header=None)
df = pd.read_csv("backend/psl_gesture_dataset.csv", header=None)

# Drop rows with any missing values (NaNs)
df = df.dropna()

# Separate features and labels
X = df.iloc[:, :-1].astype(float)  # 126 features
y = df.iloc[:, -1].astype(str)     # Labels like 'A', 'B', etc.

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Show accuracy
print("✅ Accuracy:", model.score(X_test, y_test))

# Save trained model
joblib.dump(model, "backend/psl_gesture_model.pkl")
print("✅ Model saved to backend/psl_gesture_model.pkl")
