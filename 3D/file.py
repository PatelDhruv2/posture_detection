from pathlib import Path

import pandas as pd
import plotly.express as px

# Load your CSV (resolve relative to this file, not the working directory)
csv_path = (Path(__file__).resolve().parent / ".." / "final.csv").resolve()
df = pd.read_csv(csv_path)

# Convert risk to numeric (optional for color scaling)
risk_map = {
    "LOW RISK": 1,
    "MEDIUM RISK": 2,
    "HIGH RISK": 3
}

df["RiskLevel"] = df["Risk"].map(risk_map)

# 3D Scatter Plot
fig = px.scatter_3d(
    df,
    x="RangeOfMotion",
    y="MotionSpeed",
    z="LDH_Probability",
    color="Risk",
    size="LDH_Probability",
    title="3D Spine Risk Visualization",
)

fig.show()
