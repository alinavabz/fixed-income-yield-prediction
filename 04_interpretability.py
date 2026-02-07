"""
04_interpretability.py
======================
SHAP-based model interpretability for fixed income yield prediction.

Generates:
    - Global feature importance (which macro indicators matter most)
    - SHAP summary plot (direction and magnitude of feature effects)
    - Feature dependence plots (how specific indicators affect predictions)

This is critical for communicating model logic to portfolio managers
and analysts who need to understand WHY the model makes a prediction,
not just what it predicts.
"""

import pandas as pd
import numpy as np
import json
import joblib
from pathlib import Path

import shap
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from utils import load_config, ensure_dirs, print_section


def train_final_model(X: pd.DataFrame, y: pd.Series, config: dict):
    """
    Train the best model on all data for SHAP analysis.

    In practice, you would retrain on all available data using
    the best hyperparameters found during CV.
    """
    print_section("Training Final Model for Interpretability")

    # Load best model config
    results_path = Path(config["output"]["results_path"])
    best_model_path = results_path / "best_model.json"

    if best_model_path.exists():
        with open(best_model_path) as f:
            best_info = json.load(f)
        print(f"  Best model from CV: {best_info['model_name']}")
        print(f"  Best params: {best_info.get('best_params', 'N/A')}")
    else:
        print("  No CV results found. Using XGBoost with default params.")
        best_info = {"model_name": "XGBoost"}

    seed = config["model"]["random_state"]

    # Scale features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X), columns=X.columns, index=X.index
    )

    # Feature selection
    fs_config = config["model"]["feature_selection"]
    max_features = fs_config["max_features"]

    if X_scaled.shape[1] > max_features:
        scores = mutual_info_classif(X_scaled, y, random_state=42)
        top_features = pd.Series(scores, index=X.columns).nlargest(max_features).index.tolist()
        X_final = X_scaled[top_features]
    else:
        X_final = X_scaled
        top_features = list(X.columns)

    print(f"  Using {len(top_features)} features")

    # Train model
    model_name = best_info["model_name"]

    if model_name == "XGBoost":
        model = XGBClassifier(
            random_state=seed,
            eval_metric="logloss",
            use_label_encoder=False,
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
        )
    elif model_name == "RandomForest":
        model = RandomForestClassifier(
            random_state=seed,
            n_estimators=200,
            max_depth=5,
            class_weight="balanced",
        )
    else:
        model = LogisticRegression(
            random_state=seed,
            max_iter=2000,
            C=0.1,
        )

    model.fit(X_final.values, y.values)
    print(f"  ✓ Trained {model_name} on {len(X_final)} samples")

    return model, X_final, top_features, model_name


def generate_shap_analysis(model, X: pd.DataFrame, feature_names: list,
                           model_name: str, config: dict):
    """
    Generate SHAP explanations and plots.
    """
    print_section("Generating SHAP Explanations")

    fig_path = Path(config["output"]["figures_path"])

    # Create SHAP explainer
    if model_name in ["XGBoost", "RandomForest"]:
        explainer = shap.TreeExplainer(model)
    else:
        explainer = shap.LinearExplainer(model, X.values)

    shap_values = explainer.shap_values(X.values)

    # For binary classification, shap_values might be a list
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # Class 1 (yield goes up)

    print(f"  SHAP values shape: {shap_values.shape}")

    # === Plot 1: Global Feature Importance (Bar) ===
    plt.figure(figsize=(10, 8))
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    feature_importance = pd.Series(mean_abs_shap, index=feature_names).sort_values(ascending=True)

    # Plot top 15
    top_n = min(15, len(feature_importance))
    feature_importance.tail(top_n).plot(kind="barh", color="#2563eb")
    plt.xlabel("Mean |SHAP value|")
    plt.title("Feature Importance for Yield Direction Prediction")
    plt.tight_layout()
    plt.savefig(fig_path / "shap_feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ Saved feature importance bar chart")

    # === Plot 2: SHAP Summary (Beeswarm) ===
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values,
        X.values,
        feature_names=feature_names,
        max_display=15,
        show=False,
    )
    plt.title("SHAP Summary: Impact on Yield Direction Prediction")
    plt.tight_layout()
    plt.savefig(fig_path / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ Saved SHAP summary (beeswarm) plot")

    # === Plot 3: Top 4 Dependence Plots ===
    top_4_features = feature_importance.tail(4).index.tolist()[::-1]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for idx, feat in enumerate(top_4_features):
        ax = axes[idx // 2][idx % 2]
        feat_idx = feature_names.index(feat)

        ax.scatter(
            X.values[:, feat_idx],
            shap_values[:, feat_idx],
            alpha=0.5,
            s=10,
            c="#2563eb",
        )
        ax.set_xlabel(feat)
        ax.set_ylabel("SHAP value")
        ax.set_title(f"Dependence: {feat}")
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax.grid(alpha=0.2)

    plt.suptitle("SHAP Dependence Plots — Top 4 Features", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(fig_path / "shap_dependence.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ Saved SHAP dependence plots")

    # === Save feature importance to CSV ===
    results_path = Path(config["output"]["results_path"])
    importance_df = feature_importance.reset_index()
    importance_df.columns = ["feature", "mean_abs_shap"]
    importance_df = importance_df.sort_values("mean_abs_shap", ascending=False)
    importance_df.to_csv(results_path / "feature_importance.csv", index=False)
    print(f"  ✓ Saved feature importance to CSV")

    # === Print top 10 insights ===
    print_section("Top 10 Most Important Macro Indicators")
    for rank, (_, row) in enumerate(importance_df.head(10).iterrows(), 1):
        print(f"  {rank:2d}. {row['feature']:35s} (SHAP: {row['mean_abs_shap']:.4f})")


def main():
    config = load_config()
    ensure_dirs(config)

    # Load features
    df = pd.read_csv(config["data"]["features_path"], index_col="date", parse_dates=True)
    X = df.drop(columns=["target"])
    y = df["target"]

    # Train final model
    model, X_final, feature_names, model_name = train_final_model(X, y, config)

    # SHAP analysis
    generate_shap_analysis(model, X_final, feature_names, model_name, config)

    print_section("Done")
    print("  All interpretability outputs saved to:")
    print(f"    Figures: {config['output']['figures_path']}")
    print(f"    Data:    {config['output']['results_path']}")


if __name__ == "__main__":
    main()
