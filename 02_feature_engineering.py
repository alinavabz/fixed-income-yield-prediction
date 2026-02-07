"""
02_feature_engineering.py
=========================
Build predictive features from raw macroeconomic data.

Features include:
    - Yield curve indicators (term spread, slope changes)
    - Momentum features (rate of change over multiple horizons)
    - Volatility features (rolling standard deviation)
    - Macro regime indicators (real rate, inflation momentum)
    - Market sentiment (VIX level, S&P 500 momentum)

Target:
    Binary label: 1 if 10Y yield increases next month, 0 otherwise
"""

import pandas as pd
import numpy as np
from utils import load_config, ensure_dirs, print_section


def create_target(df: pd.DataFrame, horizon: int = 1) -> pd.Series:
    """
    Create binary target: 1 if 10Y yield increases over next `horizon` months.

    Args:
        df: DataFrame with treasury_10y column
        horizon: Number of months ahead to predict

    Returns:
        Binary series (1 = yield goes up, 0 = yield goes down)
    """
    future_yield = df["treasury_10y"].shift(-horizon)
    target = (future_yield > df["treasury_10y"]).astype(int)
    return target


def create_yield_curve_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Features derived from the yield curve shape and dynamics.
    """
    features = pd.DataFrame(index=df.index)

    # Term spread: 10Y - 2Y (yield curve slope)
    features["term_spread_10y_2y"] = df["treasury_10y"] - df["treasury_2y"]

    # Term spread momentum: is the curve flattening or steepening?
    for window in [3, 6]:
        features[f"term_spread_chg_{window}m"] = features["term_spread_10y_2y"].diff(window)

    # Yield curve inversion flag (10Y < 2Y)
    features["curve_inverted"] = (features["term_spread_10y_2y"] < 0).astype(int)

    # Fed Funds vs 10Y (monetary policy tightness)
    features["ff_10y_spread"] = df["fed_funds_rate"] - df["treasury_10y"]

    return features


def create_momentum_features(df: pd.DataFrame, windows: list) -> pd.DataFrame:
    """
    Rate and price momentum features over multiple horizons.
    """
    features = pd.DataFrame(index=df.index)

    # 10Y yield momentum
    for w in windows:
        features[f"yield_10y_chg_{w}m"] = df["treasury_10y"].diff(w)
        features[f"yield_10y_pct_chg_{w}m"] = df["treasury_10y"].pct_change(w)

    # 2Y yield momentum
    for w in windows:
        features[f"yield_2y_chg_{w}m"] = df["treasury_2y"].diff(w)

    # Fed Funds Rate momentum
    for w in windows:
        features[f"ff_rate_chg_{w}m"] = df["fed_funds_rate"].diff(w)

    # S&P 500 momentum (equity market signal)
    if "sp500" in df.columns:
        for w in windows:
            features[f"sp500_return_{w}m"] = df["sp500"].pct_change(w)

    # Credit spread momentum
    if "credit_spread_baa" in df.columns:
        for w in windows:
            features[f"credit_spread_chg_{w}m"] = df["credit_spread_baa"].diff(w)

    return features


def create_volatility_features(df: pd.DataFrame, windows: list) -> pd.DataFrame:
    """
    Rolling volatility and regime features.
    """
    features = pd.DataFrame(index=df.index)

    # 10Y yield rolling volatility
    for w in windows:
        features[f"yield_10y_vol_{w}m"] = df["treasury_10y"].rolling(w).std()

    # VIX level and changes (market fear gauge)
    if "vix" in df.columns:
        features["vix_level"] = df["vix"]
        features["vix_above_20"] = (df["vix"] > 20).astype(int)
        features["vix_above_30"] = (df["vix"] > 30).astype(int)
        for w in [3, 6]:
            features[f"vix_chg_{w}m"] = df["vix"].diff(w)

    # Credit spread volatility (risk appetite)
    if "credit_spread_baa" in df.columns:
        for w in windows:
            features[f"credit_spread_vol_{w}m"] = df["credit_spread_baa"].rolling(w).std()

    return features


def create_macro_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Macroeconomic regime and inflation features.
    """
    features = pd.DataFrame(index=df.index)

    # CPI year-over-year change (inflation proxy)
    if "cpi" in df.columns:
        features["cpi_yoy"] = df["cpi"].pct_change(12) * 100

        # Inflation momentum
        features["cpi_yoy_3m_chg"] = features["cpi_yoy"].diff(3)

        # Real yield (10Y nominal minus inflation)
        features["real_yield_10y"] = df["treasury_10y"] - features["cpi_yoy"]

    # Unemployment rate and changes
    if "unemployment_rate" in df.columns:
        features["unemployment"] = df["unemployment_rate"]
        features["unemployment_chg_3m"] = df["unemployment_rate"].diff(3)
        features["unemployment_chg_6m"] = df["unemployment_rate"].diff(6)

    # Consumer sentiment
    if "consumer_sentiment" in df.columns:
        features["consumer_sentiment"] = df["consumer_sentiment"]
        features["sentiment_chg_3m"] = df["consumer_sentiment"].diff(3)

    return features


def create_level_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Raw level features and moving averages.
    """
    features = pd.DataFrame(index=df.index)

    # Current yield level
    features["yield_10y_level"] = df["treasury_10y"]

    # Moving averages (mean reversion signals)
    for w in [6, 12]:
        features[f"yield_10y_ma_{w}m"] = df["treasury_10y"].rolling(w).mean()

    # Distance from moving average (mean reversion signal)
    features["yield_10y_vs_ma12"] = (
        df["treasury_10y"] - df["treasury_10y"].rolling(12).mean()
    )

    # Credit spread level
    if "credit_spread_baa" in df.columns:
        features["credit_spread_level"] = df["credit_spread_baa"]

    # Term spread from FRED (10Y - 3M)
    if "term_spread_10y3m" in df.columns:
        features["term_spread_10y_3m"] = df["term_spread_10y3m"]

    return features


def main():
    config = load_config()
    ensure_dirs(config)
    windows = config["feature_engineering"]["rolling_windows"]
    horizon = config["data"]["target_horizon_months"]

    print_section("Loading Raw Data")
    df = pd.read_csv(config["data"]["raw_data_path"], index_col="date", parse_dates=True)
    print(f"  Loaded {len(df)} months, {len(df.columns)} raw columns")

    # Create target
    print_section("Creating Target Variable")
    target = create_target(df, horizon=horizon)
    print(f"  Target: yield direction {horizon} month(s) ahead")
    print(f"  Class distribution:\n    Up:   {(target == 1).sum()}")
    print(f"    Down: {(target == 0).sum()}")
    print(f"    Ratio: {(target == 1).sum() / (target == 0).sum():.2f}")

    # Build all feature groups
    print_section("Engineering Features")

    feature_groups = {
        "yield_curve": create_yield_curve_features(df),
        "momentum": create_momentum_features(df, windows),
        "volatility": create_volatility_features(df, windows),
        "macro": create_macro_features(df),
        "levels": create_level_features(df),
    }

    # Combine all features
    all_features = pd.concat(feature_groups.values(), axis=1)
    all_features["target"] = target

    # Report feature counts per group
    for name, group in feature_groups.items():
        print(f"  {name}: {len(group.columns)} features")
    print(f"  TOTAL: {len(all_features.columns) - 1} features")

    # Drop rows with NaN (from rolling windows and target shift)
    n_before = len(all_features)
    all_features = all_features.dropna()
    n_after = len(all_features)
    print(f"\n  Dropped {n_before - n_after} rows with NaN")
    print(f"  Final dataset: {n_after} months ({all_features.index.min().date()} to {all_features.index.max().date()})")

    # Final class distribution
    print(f"\n  Final class distribution:")
    print(f"    Up:   {(all_features['target'] == 1).sum()} ({(all_features['target'] == 1).mean():.1%})")
    print(f"    Down: {(all_features['target'] == 0).sum()} ({(all_features['target'] == 0).mean():.1%})")

    # Save
    output_path = config["data"]["features_path"]
    all_features.to_csv(output_path)
    print(f"\n  ✓ Saved to {output_path}")


if __name__ == "__main__":
    main()
