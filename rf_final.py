#!/usr/bin/env python
import argparse
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train/test a Random Forest on final.csv and save related graphs."
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to CSV file (default: final.csv next to this script).",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="Risk",
        help="Target column name (default: Risk).",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test size fraction (default: 0.2).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=300,
        help="Number of trees (default: 300).",
    )
    parser.add_argument(
        "--cv",
        type=int,
        default=5,
        help="Cross-validation folds for tuning/curves (default: 5).",
    )
    parser.add_argument(
        "--search-iter",
        type=int,
        default=30,
        help="Randomized search iterations (default: 30).",
    )
    parser.add_argument(
        "--no-tune",
        action="store_true",
        help="Disable hyperparameter tuning.",
    )
    parser.add_argument(
        "--class-weight",
        type=str,
        default="balanced",
        help="Class weight setting (default: balanced). Use 'none' to disable.",
    )
    return parser.parse_args()


def resolve_data_path(arg_path: str | None) -> Path:
    if arg_path:
        return Path(arg_path)
    return Path(__file__).resolve().parent / "final.csv"


def main() -> None:
    args = parse_args()
    data_path = resolve_data_path(args.data)

    if not data_path.exists():
        raise FileNotFoundError(f"CSV not found: {data_path}")

    df = pd.read_csv(data_path)

    if args.target not in df.columns:
        raise ValueError(
            f"Target column '{args.target}' not found. Columns: {list(df.columns)}"
        )

    X = df.drop(columns=[args.target])
    y = df[args.target]

    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = [c for c in X.columns if c not in categorical_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", "passthrough", numeric_cols),
        ],
        remainder="drop",
    )

    class_weight = None if args.class_weight.lower() == "none" else args.class_weight
    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        random_state=args.random_state,
        n_jobs=-1,
        class_weight=class_weight,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y if y.nunique() > 1 else None,
    )

    clf = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])

    if args.no_tune:
        best_clf = clf.fit(X_train, y_train)
        best_params = None
        best_cv = None
    else:
        param_distributions = {
            "model__n_estimators": [200, 300, 400, 600, 800],
            "model__max_depth": [None, 5, 8, 12, 16, 20, 24],
            "model__min_samples_split": [2, 4, 6, 8],
            "model__min_samples_leaf": [1, 2, 3, 4],
            "model__max_features": ["sqrt", "log2", 0.5, 0.8],
            "model__bootstrap": [True, False],
        }
        search = RandomizedSearchCV(
            clf,
            param_distributions=param_distributions,
            n_iter=args.search_iter,
            cv=args.cv,
            scoring="accuracy",
            random_state=args.random_state,
            n_jobs=-1,
            refit=True,
        )
        search.fit(X_train, y_train)
        best_clf = search.best_estimator_
        best_params = search.best_params_
        best_cv = search.best_score_

    y_pred = best_clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()
