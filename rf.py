from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "flex_dataset.csv"
MODEL_PATH = BASE_DIR / "flex_rf_model.pkl"
PLOT_PATH = BASE_DIR / "test_posture_comparison.png"

# Load dataset
df = pd.read_csv(DATA_PATH)

# Features and label
X = df.drop("label", axis=1)
y = df["label"]

# Split 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train RF
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=42
)
model.fit(X_train, y_train)

# Predict + accuracy
pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred))

# Save model
joblib.dump(model, MODEL_PATH)

# Plot actual vs predicted posture for each test row
y_test = y_test.reset_index(drop=True)
pred_s = pd.Series(pred, name="predicted")

classes = sorted(y.unique())
label_to_id = {label: i for i, label in enumerate(classes)}

actual_ids = y_test.map(label_to_id)
pred_ids = pred_s.map(label_to_id)

plt.figure(figsize=(12, 5))
x = range(len(y_test))
plt.plot(x, actual_ids, marker="o", linewidth=1.8, label="Actual posture")
plt.plot(x, pred_ids, marker="x", linewidth=1.8, label="Predicted posture")

plt.yticks(list(label_to_id.values()), list(label_to_id.keys()))
plt.xlabel("Test row index")
plt.ylabel("Posture")
plt.title("Actual vs Predicted Posture (Test Rows)")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(PLOT_PATH, dpi=200)
plt.show()

print(f"Saved plot to: {PLOT_PATH}")
