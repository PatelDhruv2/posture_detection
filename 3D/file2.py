from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
csv_path = (Path(__file__).resolve().parent / ".." / "final.csv").resolve()
df = pd.read_csv(csv_path)
x = np.linspace(df["RangeOfMotion"].min(), df["RangeOfMotion"].max(), 30)
y = np.linspace(df["MotionSpeed"].min(), df["MotionSpeed"].max(), 30)

X, Y = np.meshgrid(x, y)
Z = (X / X.max()) * 50 + (Y / Y.max()) * 50

fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y)])

fig.update_layout(
    title="Risk Surface (Approximation)",
    scene=dict(
        xaxis_title="Range of Motion",
        yaxis_title="Motion Speed",
        zaxis_title="Risk"
    )
)

fig.show()
