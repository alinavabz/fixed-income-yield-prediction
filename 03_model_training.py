"""
03_model_training.py
====================
Train and evaluate ML models using time-series cross-validation.

Key design decisions:
    - Expanding window CV (never train on future data)
    - Gap between train and test to prevent leakage
    - Hyperparameter tuning via inner CV loop
    - Multiple models compared on same folds

This is the financial equivalent of nested cross-validation,
adapted for temporal data where random shuffling is invalid.
"""

import pandas as pd
import numpy as np
import json
import joblib
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif, SelectKBest, f_classif
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
)

import matplotlib.pyplot as plt
import seaborn as sns

from utils import load_config, ensure_dirs, print_section


class TimeSeriesExpandingCV:
    """
    Expanding window cross-validation for financial time series.

    Unlike standard k-fold, this ensures:
        - Training data is always BEFORE test data
        - A gap between train and test prevents information leakage
        - Training window expands with each fold (more data over time)
    """

    def __init__(self, n_splits: int, min_train_size: int, gap: int = 1):
        self.n_splits = n_splits
        self.min_train_size = min_train_size
        self.gap = gap

    def split(self, X):
        n_samples = len(X)
        test_size = (n_samples - self.min_train_size - self.gap * self.n_splits) // self.n_splits

        if test_size < 10:
            raise ValueError(
                f"Not enough data for {self.n_splits} splits. "
                f"Have {n_samples} samples, need at least "
                f"{self.min_train_size + (test_size + self.gap) * self.n_splits}"
            )

        indices = np.arange(n_samples)
        splits = []

        for i in range(self.n_splits):
            test_start = self.min_train_size + self.gap + i * (test_size + self.gap)
            test_end = test_start + test_size

            if test_end > n_samples:
                test_end = n_samples

            train_end = test_start - self.gap
            train_indices = indices[:train_end]
            test_indices = indices[test_start:test_end]

            if len(test_indices) > 0:
                splits.append((train_indices, test_indices))

        return splits


def get_models(config: dict) -> dict:
    """
    Initialize models with their hyperparameter grids.
    """
    model_configs = config["model"]["models"]
    seed = config["model"]["random_state"]

    models = {}

    if "logistic_regression" in model_configs:
        models["LogisticRegression"] = {
            "model": LogisticRegression(random_state=seed, max_iter=2000),
            "params": model_configs["logistic_regression"]["params"],
        }

    if "random_forest" in model_configs:
        models["RandomForest"] = {
            "model": RandomForestClassifier(random_state=seed),
            "params": model_configs["random_forest"]["params"],
        }

    if "xgboost" in model_configs:
        models["XGBoost"] = {
            "model": XGBClassifier(
                random_state=seed,
                eval_metric="logloss",
                use_label_encoder=False,
            ),
            "params": model_configs["xgboost"]["params"],
        }

    return models


def select_features(X_train, y_train, X_test, max_features: int, method: str):
    """
    Select top features based on training data only.

    Args:
        X_train, y_train: Training data
        X_test: Test data (transformed, not fitted)
        max_features: Number of features to keep
        method: 'mutual_info', 'f_classif', or 'none'

    Returns:
        X_train_selected, X_test_selected, selected_feature_names
    """
    if method == "none" or X_train.shape[1] <= max_features:
        return X_train, X_test, list(X_train.columns)

    if method == "mutual_info":
        scores = mutual_info_classif(X_train, y_train, random_state=42)
    elif method == "f_classif":
        scores, _ = f_classif(X_train, y_train)
    else:
        raise ValueError(f"Unknown feature selection method: {method}")

    # Rank and select top features
    feature_scores = pd.Series(scores, index=X_train.columns)
    top_features = feature_scores.nlargest(max_features).index.tolist()

    return X_train[top_features], X_test[top_features], top_features


def train_and_evaluate(config: dict):
    """
    Main training loop with time-series CV.
    """
    print_section("Loading Features")
    df = pd.read_csv(config["data"]["features_path"], index_col="date", parse_dates=True)

    X = df.drop(columns=["target"])
    y = df["target"]
    feature_names = list(X.columns)

    print(f"  Samples: {len(X)}")
    print(f"  Features: {len(feature_names)}")
    print(f"  Date range: {X.index.min().date()} to {X.index.max().date()}")

    # Set up time-series CV
    cv_config = config["model"]["cv"]
    cv = TimeSeriesExpandingCV(
        n_splits=cv_config["n_splits"],
        min_train_size=cv_config["min_train_months"],
        gap=cv_config["gap_months"],
    )
    splits = cv.split(X)

    # Get models
    models = get_models(config)
    fs_config = config["model"]["feature_selection"]

    # Storage
    all_results = []
    best_overall = {"auc": 0, "model_name": None, "fold": None}

    print_section("Running Time-Series Cross-Validation")

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        print(f"\n{'─' * 50}")
        print(f"  FOLD {fold_idx + 1}/{len(splits)}")
        print(f"{'─' * 50}")

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        print(f"  Train: {X_train.index.min().date()} to {X_train.index.max().date()} ({len(X_train)} months)")
        print(f"  Test:  {X_test.index.min().date()} to {X_test.index.max().date()} ({len(X_test)} months)")
        print(f"  Train class balance: {y_train.mean():.2%} positive")

        # Scale features (fit on train only)
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=feature_names,
            index=X_train.index,
        )
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test),
            columns=feature_names,
            index=X_test.index,
        )

        # Feature selection (on train only)
        X_train_fs, X_test_fs, selected_features = select_features(
            X_train_scaled,
            y_train,
            X_test_scaled,
            max_features=fs_config["max_features"],
            method=fs_config["method"],
        )
        print(f"  Selected {len(selected_features)} features")

        # Train each model
        for model_name, model_info in models.items():
            # Inner CV for hyperparameter tuning (also time-series aware)
            inner_cv = TimeSeriesSplit(n_splits=3)

            grid = GridSearchCV(
                model_info["model"],
                model_info["params"],
                cv=inner_cv,
                scoring="roc_auc",
                n_jobs=-1,
                refit=True,
            )

            grid.fit(X_train_fs.values, y_train.values)

            # Predict on test set
            y_pred = grid.predict(X_test_fs.values)
            y_prob = grid.predict_proba(X_test_fs.values)[:, 1]

            # Metrics
            auc = roc_auc_score(y_test, y_prob)
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred)

            # Track best
            if auc > best_overall["auc"]:
                best_overall = {
                    "auc": auc,
                    "model_name": model_name,
                    "fold": fold_idx + 1,
                    "best_params": grid.best_params_,
                }

            result = {
                "fold": fold_idx + 1,
                "model": model_name,
                "auc": round(auc, 4),
                "accuracy": round(acc, 4),
                "f1": round(f1, 4),
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "best_params": str(grid.best_params_),
                "n_features": len(selected_features),
                "inner_auc": round(grid.best_score_, 4),
                "overfit_gap": round(grid.best_score_ - auc, 4),
                "train_start": str(X_train.index.min().date()),
                "train_end": str(X_train.index.max().date()),
                "test_start": str(X_test.index.min().date()),
                "test_end": str(X_test.index.max().date()),
            }
            all_results.append(result)

            print(f"    {model_name:25s} | AUC: {auc:.3f} | Acc: {acc:.3f} | F1: {f1:.3f} | Overfit gap: {grid.best_score_ - auc:+.3f}")

    return all_results, best_overall


def print_summary(results: list, best_overall: dict, config: dict):
    """Print and save results summary."""
    print_section("FINAL RESULTS")

    df_results = pd.DataFrame(results)

    # Per-model summary
    summary = df_results.groupby("model").agg(
        auc_mean=("auc", "mean"),
        auc_std=("auc", "std"),
        acc_mean=("accuracy", "mean"),
        acc_std=("accuracy", "std"),
        f1_mean=("f1", "mean"),
        f1_std=("f1", "std"),
        overfit_mean=("overfit_gap", "mean"),
    ).round(4)

    print("  Model Comparison (mean ± std across folds):\n")
    for model_name, row in summary.iterrows():
        print(f"  {model_name:25s}")
        print(f"    AUC:      {row['auc_mean']:.3f} ± {row['auc_std']:.3f}")
        print(f"    Accuracy: {row['acc_mean']:.3f} ± {row['acc_std']:.3f}")
        print(f"    F1:       {row['f1_mean']:.3f} ± {row['f1_std']:.3f}")
        print(f"    Overfit:  {row['overfit_mean']:+.3f}")
        print()

    print(f"\n  Best overall: {best_overall['model_name']} "
          f"(Fold {best_overall['fold']}, AUC={best_overall['auc']:.3f})")

    # Save results
    results_path = Path(config["output"]["results_path"])
    df_results.to_csv(results_path / "cv_results.csv", index=False)
    summary.to_csv(results_path / "model_comparison.csv")

    with open(results_path / "best_model.json", "w") as f:
        json.dump(best_overall, f, indent=2, default=str)

    print(f"\n  ✓ Results saved to {results_path}")


def plot_results(results: list, config: dict):
    """Generate result visualizations."""
    df_results = pd.DataFrame(results)
    fig_path = Path(config["output"]["figures_path"])

    # AUC comparison across folds
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Box plot of AUC by model
    model_names = df_results["model"].unique()
    auc_data = [df_results[df_results["model"] == m]["auc"].values for m in model_names]
    axes[0].boxplot(auc_data, labels=model_names)
    axes[0].set_ylabel("AUC")
    axes[0].set_title("AUC Distribution Across Folds")
    axes[0].axhline(y=0.5, color="r", linestyle="--", alpha=0.5, label="Random baseline")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.3)

    # Fold-by-fold AUC
    for model_name in model_names:
        model_data = df_results[df_results["model"] == model_name]
        axes[1].plot(model_data["fold"], model_data["auc"], "o-", label=model_name)
    axes[1].set_xlabel("Fold")
    axes[1].set_ylabel("AUC")
    axes[1].set_title("AUC by Fold (Expanding Window)")
    axes[1].axhline(y=0.5, color="r", linestyle="--", alpha=0.5)
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(fig_path / "model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved model comparison plot")


def main():
    config = load_config()
    ensure_dirs(config)

    results, best_overall = train_and_evaluate(config)
    print_summary(results, best_overall, config)
    plot_results(results, config)


if __name__ == "__main__":
    main()
