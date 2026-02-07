"""
01_data_collection.py
=====================
Pull macroeconomic and market data from FRED and Yahoo Finance.

Usage:
    export FRED_API_KEY=your_key_here
    python 01_data_collection.py

Data Sources:
    - FRED: Treasury yields, Fed Funds Rate, CPI, unemployment,
            credit spreads, term spreads, consumer sentiment
    - Yahoo Finance: VIX, S&P 500
"""

import os
import pandas as pd
import numpy as np
from fredapi import Fred
import yfinance as yf
from utils import load_config, ensure_dirs, resample_to_monthly, print_section


def fetch_fred_data(config: dict, fred: Fred) -> pd.DataFrame:
    """
    Fetch all FRED series and combine into a single DataFrame.

    Args:
        config: Project configuration dict
        fred: Authenticated Fred API client

    Returns:
        DataFrame with monthly macroeconomic indicators
    """
    print_section("Fetching FRED Data")

    start = config["data"]["start_date"]
    end = config["data"]["end_date"]
    series_list = config["data"]["fred_series"]

    frames = {}
    for series_info in series_list:
        series_id = series_info["id"]
        name = series_info["name"]
        desc = series_info["description"]

        print(f"  Fetching {series_id} ({desc})...")

        try:
            data = fred.get_series(series_id, start, end)
            frames[name] = data
            print(f"    ✓ {len(data)} observations")
        except Exception as e:
            print(f"    ✗ Failed: {e}")

    # Combine all series
    df_fred = pd.DataFrame(frames)
    df_fred.index = pd.to_datetime(df_fred.index)
    df_fred.index.name = "date"

    return df_fred


def fetch_yahoo_data(config: dict) -> pd.DataFrame:
    """
    Fetch market data from Yahoo Finance.

    Args:
        config: Project configuration dict

    Returns:
        DataFrame with monthly market indicators
    """
    print_section("Fetching Yahoo Finance Data")

    start = config["data"]["start_date"]
    end = config["data"]["end_date"]
    tickers = config["data"]["yahoo_tickers"]

    frames = {}
    for ticker_info in tickers:
        ticker = ticker_info["ticker"]
        name = ticker_info["name"]
        desc = ticker_info["description"]

        print(f"  Fetching {ticker} ({desc})...")

        try:
            data = yf.download(ticker, start=start, end=end, progress=False)
            # Use Adjusted Close for price series
            if "Adj Close" in data.columns:
                frames[name] = data["Adj Close"]
            else:
                frames[name] = data["Close"]
            print(f"    ✓ {len(data)} observations")
        except Exception as e:
            print(f"    ✗ Failed: {e}")

    df_yahoo = pd.DataFrame(frames)
    df_yahoo.index = pd.to_datetime(df_yahoo.index)
    df_yahoo.index.name = "date"

    return df_yahoo


def combine_and_clean(df_fred: pd.DataFrame, df_yahoo: pd.DataFrame) -> pd.DataFrame:
    """
    Resample to monthly, merge, and handle missing values.

    Args:
        df_fred: FRED data (potentially daily)
        df_yahoo: Yahoo Finance data (daily)

    Returns:
        Clean monthly DataFrame
    """
    print_section("Combining and Cleaning Data")

    # Resample both to monthly
    fred_monthly = resample_to_monthly(df_fred)
    yahoo_monthly = resample_to_monthly(df_yahoo)

    # Merge on date index
    df = fred_monthly.join(yahoo_monthly, how="outer")

    # Forward-fill missing values (common for macro data with different release dates)
    # Then backward-fill any remaining NaNs at the start
    df = df.ffill().bfill()

    # Drop rows where critical columns are still NaN
    critical_cols = ["treasury_10y"]
    df = df.dropna(subset=critical_cols)

    print(f"  Date range: {df.index.min().date()} to {df.index.max().date()}")
    print(f"  Months: {len(df)}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Missing values:\n{df.isna().sum()}")

    return df


def main():
    # Load config
    config = load_config()
    ensure_dirs(config)

    # Authenticate with FRED
    api_key = os.environ.get("FRED_API_KEY")
    if not api_key:
        raise ValueError(
            "FRED_API_KEY environment variable not set.\n"
            "Get a free key at: https://fred.stlouisfed.org/docs/api/api_key.html\n"
            "Then run: export FRED_API_KEY=your_key_here"
        )

    fred = Fred(api_key=api_key)

    # Fetch data
    df_fred = fetch_fred_data(config, fred)
    df_yahoo = fetch_yahoo_data(config)

    # Combine and clean
    df = combine_and_clean(df_fred, df_yahoo)

    # Save
    output_path = config["data"]["raw_data_path"]
    df.to_csv(output_path)
    print(f"\n  ✓ Saved to {output_path} ({len(df)} rows, {len(df.columns)} columns)")


if __name__ == "__main__":
    main()
