# ai/ml/generate_flex_dataset.py
from pathlib import Path
import argparse
import numpy as np
import pandas as pd

COLUMNS = [
    "Mean",
    "Standard deviation",
    "Max",
    "Min",
    "RMS",
    "Energy",
    "Peak-to-peak",
    "Zero-crossing rate",
    "label",
]

LABEL_CONFIG = {
    "BENDING": {
        "mean_mu": 78, "mean_sigma": 12,
        "std_mu": 10, "std_sigma": 3,
        "zcr_mu": 0.14, "zcr_sigma": 0.03,
    },
    "STRAIGHT": {
        "mean_mu": 28, "mean_sigma": 7,
        "std_mu": 5, "std_sigma": 1.8,
        "zcr_mu": 0.065, "zcr_sigma": 0.015,
    },
    "RETURNING": {
        "mean_mu": 52, "mean_sigma": 10,
        "std_mu": 8, "std_sigma": 2.5,
        "zcr_mu": 0.11, "zcr_sigma": 0.025,
    },
}

def generate_row(label: str, rng: np.random.Generator) -> dict:
    cfg = LABEL_CONFIG[label]

    mean = rng.normal(cfg["mean_mu"], cfg["mean_sigma"])
    std = max(1.2, rng.normal(cfg["std_mu"], cfg["std_sigma"]))

    # Create realistic min/max around mean
    low_span = abs(rng.normal(1.3 * std, 0.5 * std))
    high_span = abs(rng.normal(1.7 * std, 0.6 * std))
    min_v = mean - low_span
    max_v = mean + high_span

    # Keep signal-like relationships
    p2p = max_v - min_v
    rms = np.sqrt(mean**2 + std**2) * rng.uniform(0.96, 1.04)
    energy = (rms**2) * rng.uniform(0.22, 0.48)
    zcr = np.clip(rng.normal(cfg["zcr_mu"], cfg["zcr_sigma"]), 0.01, 0.30)

    return {
        "Mean": round(mean, 3),
        "Standard deviation": round(std, 3),
        "Max": round(max_v, 3),
        "Min": round(min_v, 3),
        "RMS": round(rms, 3),
        "Energy": round(energy, 3),
        "Peak-to-peak": round(p2p, 3),
        "Zero-crossing rate": round(zcr, 3),
        "label": label,
    }

def build_dataset(n_rows: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    labels = np.array(list(LABEL_CONFIG.keys()))
    probs = np.array([0.4, 0.3, 0.3])  # BENDING, STRAIGHT, RETURNING
    chosen = rng.choice(labels, size=n_rows, p=probs)

    rows = [generate_row(lbl, rng) for lbl in chosen]
    return pd.DataFrame(rows, columns=COLUMNS)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "flex_dataset.csv",
    )
    args = parser.parse_args()

    df = build_dataset(args.rows, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Generated {len(df)} rows -> {args.out}")

if __name__ == "__main__":
    main()
