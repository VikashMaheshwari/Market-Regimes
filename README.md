# Market Regime Dashboard

Interactive Streamlit dashboard for the S&P 500 GMM/HMM regime-detection capstone project.

## Run locally

```bash
cd "C:\Users\vikm6\OneDrive\Documents\regime_dashboard"
pip install -r requirements.txt
streamlit run app.py
```

The first run takes ~30–60 seconds (downloads training data 2010–2024, fits four models). Subsequent loads are instant — models are cached in the Streamlit session, market data is cached for 24 hours.

## What it does

- **Sidebar:** Toggle between GMM / HMM and K=2 / K=3. Hit "Refresh data" to pull the latest weekly close.
- **KPI strip:** Current regime, confidence, weeks held, and OOS strategy CAGR vs Buy & Hold.
- **Current State tab:** S&P 500 chart with regime shading + per-week confidence bar.
- **Backtest tab:** Equity curve and drawdown for the regime-aware strategy vs Buy & Hold, with full performance metrics.
- **Predictions tab:** Last N weeks of regime calls, downloadable as CSV.
- **Diagnostics tab:** HMM transition matrix, expected dwell times, regime distribution, mean feature values per regime.

## Notes

- The app retrains from scratch on first load (no need to run the notebook first). Hyperparameters are hardcoded to the BIC-optimal config — `covariance_type="full"`, `reg_covar=1e-3` — to keep startup fast. To re-run the full grid search, do it in the notebook and copy in the winners.
- "Refresh data" clears the data cache and reruns; the models stay cached because retraining 4 models on every refresh would be wasteful. To force retraining, restart the Streamlit process.
