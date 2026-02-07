"""
Utility functions for Fixed Income Yield Prediction project.
"""

import os
import yaml
import pandas as pd
import numpy as np
from pathlib import Path


def load_config(config_path: str = "config.yaml") -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def ensure_dirs(config: dict) -> None:
    """Create output directories if they don't exist."""
    dirs = [
        "data",
        config["output"]["results_path"],
        config["output"]["figures_path"],
        config["output"]["model_path"],
    ]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


def resample_to_monthly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resample daily/irregular data to monthly frequency.
    Uses end-of-month values for rates, mean for indices.
    """
    # Use last available value of each month
    monthly = df.resample("ME").last()
    return monthly


def print_section(title: str) -> None:
    """Print a formatted section header."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")
