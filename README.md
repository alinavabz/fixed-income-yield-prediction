# Fixed Income Yield Prediction — Machine Learning for Treasury Markets

Predicting the **direction of US 10-Year Treasury yield movements** using macroeconomic indicators and ML models with proper time-series cross-validation to prevent look-ahead bias.

## Problem Statement

Fixed income portfolio managers need to anticipate interest rate movements for asset allocation and duration management. This project builds a binary classification pipeline that predicts whether the 10Y Treasury yield will **increase or decrease** over the next month using publicly available macroeconomic data from FRED.

## Key Results

| Model | AUC | Accuracy | F1 |
|---|---|---|---|
| Logistic Regression | Baseline | — | — |
| Random Forest | — | — | — |
| XGBoost | — | — | — |

*(Fill in after running — results depend on date range)*

## Methodology

1. **Data Collection**: 10Y/2Y Treasury yields, Fed Funds Rate, CPI, unemployment, VIX, S&P 500, credit spreads from FRED and Yahoo Finance
2. **Feature Engineering**: Term spread, yield momentum, rolling volatility, rate of change, moving averages
3. **Time-Series CV**: Expanding window cross-validation (no look-ahead bias) — critical for financial data
4. **Models**: Logistic Regression, Random Forest, XGBoost with hyperparameter tuning
5. **Interpretability**: SHAP values to identify which macro indicators drive predictions

## Project Structure

```
├── config.yaml              # All parameters (data range, features, model hyperparams)
├── 01_data_collection.py    # Pull data from FRED / Yahoo Finance
├── 02_feature_engineering.py # Build predictive features
├── 03_model_training.py     # Time-series CV + model comparison
├── 04_interpretability.py   # SHAP analysis
├── utils.py                 # Helper functions
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

You will need a free FRED API key from https://fred.stlouisfed.org/docs/api/api_key.html

```bash
export FRED_API_KEY=your_key_here
```

## Run

```bash
python 01_data_collection.py
python 02_feature_engineering.py
python 03_model_training.py
python 04_interpretability.py
```

## Skills Demonstrated

- **Python, pandas, scikit-learn, XGBoost** — end-to-end ML pipeline
- **Time-series cross-validation** — proper financial backtesting methodology
- **Feature engineering** — domain-driven features from macroeconomic data
- **SHAP interpretability** — explaining predictions to non-technical stakeholders
- **YAML configuration** — reproducible experiment management
- **Quantitative research** — systematic approach to financial prediction

## Author

Ali Navab Zadeh — MSc Computer Science, Toronto Metropolitan University
