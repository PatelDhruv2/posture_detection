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

# Class-specific signal profile for more realistic overlap/noise behavior.
POSTURE_CONFIG = {
    "BENDING": {
        "baseline_mu": 80.0,
        "baseline_sigma": 8.0,
        "amp_mu": 7.0,
        "amp_sigma": 2.0,
        "noise_mu": 3.2,
        "noise_sigma": 0.9,
        "drift_mu": 1.6,
        "drift_sigma": 0.6,
        "freq_mu": 1.6,
        "freq_sigma": 0.35,
    },
    "RETURNING": {
        "baseline_mu": 52.0,
        "baseline_sigma": 7.0,
        "amp_mu": 5.2,
        "amp_sigma": 1.7,
        "noise_mu": 2.7,
        "noise_sigma": 0.8,
        "drift_mu": 1.2,
        "drift_sigma": 0.5,
        "freq_mu": 1.35,
        "freq_sigma": 0.30,
    },
    "STRAIGHT": {
        "baseline_mu": 28.0,
        "baseline_sigma": 5.5,
        "amp_mu": 3.3,
        "amp_sigma": 1.3,
        "noise_mu": 1.9,
        "noise_sigma": 0.6,
        "drift_mu": 0.7,
        "drift_sigma": 0.35,
        "freq_mu": 1.05,
        "freq_sigma": 0.25,
    },
}


def _bounded_normal(rng: np.random.Generator, mu: float, sigma: float, lo: float) -> float:
    return max(lo, float(rng.normal(mu, sigma)))


def _simulate_window(cfg: dict, rng: np.random.Generator, n: int = 128) -> np.ndarray:
    t = np.linspace(0.0, 1.0, n, endpoint=False)

    baseline = float(rng.normal(cfg["baseline_mu"], cfg["baseline_sigma"]))
    amp = _bounded_normal(rng, cfg["amp_mu"], cfg["amp_sigma"], lo=0.6)
    noise_sigma = _bounded_normal(rng, cfg["noise_mu"], cfg["noise_sigma"], lo=0.2)
    drift_strength = _bounded_normal(rng, cfg["drift_mu"], cfg["drift_sigma"], lo=0.0)
    freq = _bounded_normal(rng, cfg["freq_mu"], cfg["freq_sigma"], lo=0.3)
    phase = float(rng.uniform(0.0, 2.0 * np.pi))

    harmonic = 0.28 * amp * np.sin(2.0 * np.pi * (2.0 * freq) * t + phase / 3.0)
    drift = drift_strength * (t - 0.5)
    noise = rng.normal(0.0, noise_sigma, size=n)
    signal = baseline + amp * np.sin(2.0 * np.pi * freq * t + phase) + harmonic + drift + noise

    # Flex sensor output is non-negative ADC-like magnitude.
    signal = np.clip(signal, 0.0, None)
    return signal


def _extract_features(signal: np.ndarray, label: str) -> dict:
    mean = float(np.mean(signal))
    std = float(np.std(signal, ddof=0))
    max_v = float(np.max(signal))
    min_v = float(np.min(signal))
    rms = float(np.sqrt(np.mean(signal ** 2)))
    energy = float(np.sum(signal ** 2) / len(signal))
    p2p = max_v - min_v

    centered = signal - mean
    zcr = float(np.mean(np.diff(np.signbit(centered)).astype(float)))

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


def generate_row(label: str, rng: np.random.Generator, window_size: int = 128) -> dict:
    sig = _simulate_window(POSTURE_CONFIG[label], rng, n=window_size)
    return _extract_features(sig, label)


def build_dataset(n_rows: int, seed: int = 123, window_size: int = 128) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    labels = np.array(list(POSTURE_CONFIG.keys()))
    probs = np.array([0.4, 0.3, 0.3])  # BENDING, RETURNING, STRAIGHT
    chosen = rng.choice(labels, size=n_rows, p=probs)

    rows = [generate_row(lbl, rng, window_size=window_size) for lbl in chosen]
    return pd.DataFrame(rows, columns=COLUMNS)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate realistic synthetic flex posture dataset.")
    parser.add_argument("--rows", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--window-size", type=int, default=128)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "flex_dataset_realistic.csv",
    )
    args = parser.parse_args()

    df = build_dataset(args.rows, seed=args.seed, window_size=args.window_size)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Generated {len(df)} rows -> {args.out}")


if __name__ == "__main__":
    main()
