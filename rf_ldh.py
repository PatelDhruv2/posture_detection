from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "posture_dataset_voltage.csv"
MODEL_PATH = BASE_DIR / "ldh_rf_model.pkl"

# Load dataset
if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)

# Expected columns
label_col = "LDH"
feature_cols = [
    "RangeOfMotion",
    "MotionSpeed",
    "PostureDuration",
    "Smoothness",
    "JerkRMS",
    "MeanVelocity",
    "MaxVelocity",
]

missing = [c for c in feature_cols + [label_col] if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns in dataset: {missing}")

# Clean labels
y = df[label_col].astype(str).str.strip()

# Drop unknown/unlabeled rows
bad = y.isna() | (y == "") | (y.str.upper() == "UNKNOWN")
if bad.any():
    df = df.loc[~bad].copy()
    y = df[label_col].astype(str).str.strip()

if df.empty:
    raise ValueError("No labeled rows found (LDH column empty or UNKNOWN).")

X = df[feature_cols]

# Split
label_counts = y.value_counts()
too_small = label_counts[label_counts < 2]
if len(too_small) > 0:
    print(
        "Warning: Some classes have <2 samples; disabling stratification.",
        "Classes:", list(too_small.index)
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=None
    )
else:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

# Train RF
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=6,
    random_state=42
)
model.fit(X_train, y_train)

# Predict + accuracy
pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred))
print("Confusion matrix:\n", confusion_matrix(y_test, pred))
print("Report:\n", classification_report(y_test, pred))

# Save model
joblib.dump({"model": model, "features": feature_cols}, MODEL_PATH)
print(f"Saved model to: {MODEL_PATH}")
