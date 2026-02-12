# ai/ml/analytics_graphs.py
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

BASE_DIR = Path(__file__).resolve().parent
csv_path = BASE_DIR / "flex_dataset.csv"
out_dir = BASE_DIR / "plots"
out_dir.mkdir(exist_ok=True)

df = pd.read_csv(csv_path)

# Separate features/label
X = df.drop(columns=["label"])
y = df["label"]

# 1) Label distribution
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="label", hue="label", legend=False)
plt.title("Label Distribution")
plt.tight_layout()
plt.savefig(out_dir / "label_distribution.png", dpi=200)
plt.close()

# 2) Feature histograms
X.hist(figsize=(12, 8), bins=20)
plt.suptitle("Feature Distributions", y=1.02)
plt.tight_layout()
plt.savefig(out_dir / "feature_histograms.png", dpi=200)
plt.close()

# 3) Correlation heatmap
plt.figure(figsize=(10, 7))
corr = X.corr(numeric_only=True)
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig(out_dir / "correlation_heatmap.png", dpi=200)
plt.close()

# 4) Boxplots by label
for col in X.columns:
    plt.figure(figsize=(7, 4))
    sns.boxplot(data=df, x="label", y=col, hue="label", legend=False)
    plt.title(f"{col} by Label")
    plt.tight_layout()
    safe_name = col.lower().replace(" ", "_").replace("-", "_")
    plt.savefig(out_dir / f"boxplot_{safe_name}.png", dpi=200)
    plt.close()

# 5) Random Forest feature importance
rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf.fit(X, y)
imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)

plt.figure(figsize=(8, 5))
sns.barplot(x=imp.values, y=imp.index, hue=imp.index, dodge=False, legend=False)
plt.title("Random Forest Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig(out_dir / "rf_feature_importance.png", dpi=200)
plt.close()

print(f"Saved plots to: {out_dir}")
