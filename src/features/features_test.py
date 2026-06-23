"""Permutation importance for handcrafted argument features."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from data_loader import load_pairs
from pairwise import build_argument_dataset, build_pairwise_dataset, get_feature_names

DEFAULT_CSV_PATH = (
    "/Users/tristanjin/Documents/TELECOM_PARIS/2A_COURS/SD/Projet-DSAI/"
    "Datasets/WinningArgCorpus/WAC.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Permutation importance for handcrafted argument features"
    )
    parser.add_argument(
        "--csv-path",
        default=DEFAULT_CSV_PATH,
        help="Path to WAC.csv",
    )
    parser.add_argument(
        "--mode",
        choices=("features", "pairwise_features", "both"),
        default="both",
        help="Which feature family to evaluate",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Hold-out fraction used for test set",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=10,
        help="Number of permutation repeats per feature",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=15,
        help="Number of features displayed in the report",
    )
    return parser.parse_args()


def load_features_dataset(csv_path: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    data = pd.read_csv(csv_path)
    X, y, feature_names = build_argument_dataset(data, include_op=True)
    return X, y, feature_names


def load_pairwise_features_dataset(
    csv_path: str,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    pairs = load_pairs(csv_path)
    X, y = build_pairwise_dataset(pairs)
    feature_names = get_feature_names()
    return X, y, feature_names


def make_estimator(mode: str) -> Pipeline:
    if mode == "features":
        model = SVC(kernel="linear", class_weight="balanced")
    elif mode == "pairwise_features":
        model = LogisticRegression(max_iter=1000, class_weight="balanced")
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", model),
        ]
    )


def evaluate_mode(
    mode: str,
    csv_path: str,
    test_size: float,
    random_state: int,
    n_repeats: int,
    top_k: int,
) -> None:
    if mode == "features":
        X, y, feature_names = load_features_dataset(csv_path)
    elif mode == "pairwise_features":
        X, y, feature_names = load_pairwise_features_dataset(csv_path)
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    estimator = make_estimator(mode)
    estimator.fit(X_train, y_train)

    y_pred = estimator.predict(X_test)
    if hasattr(estimator, "decision_function"):
        y_score = estimator.decision_function(X_test)
    else:
        y_score = estimator.predict_proba(X_test)[:, 1]

    print(f"\n=== Mode: {mode} ===")
    print(classification_report(y_test, y_pred))
    print(f"AUC: {roc_auc_score(y_test, y_score):.4f}")

    importance = permutation_importance(
        estimator,
        X_test,
        y_test,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring="roc_auc",
        n_jobs=-1,
    )

    result = pd.DataFrame(
        {
            "feature": feature_names,
            "importance_mean": importance.importances_mean,
            "importance_std": importance.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)

    print(f"\nTop {top_k} permutation importances:")
    for _, row in result.head(top_k).iterrows():
        print(
            f"- {row['feature']}: {row['importance_mean']:.6f} ± {row['importance_std']:.6f}"
        )

    out_path = CURRENT_DIR / f"permutation_importance_{mode}.csv"
    result.to_csv(out_path, index=False)
    print(f"\nSaved detailed results to: {out_path}")


def main() -> None:
    args = parse_args()
    modes = [args.mode] if args.mode != "both" else ["features", "pairwise_features"]

    for mode in modes:
        evaluate_mode(
            mode=mode,
            csv_path=args.csv_path,
            test_size=args.test_size,
            random_state=args.random_state,
            n_repeats=args.n_repeats,
            top_k=args.top_k,
        )


if __name__ == "__main__":
    main()
