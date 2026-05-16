# stock_model — multi-model next-day return prediction

A small, honest pipeline that trains several models to predict the **next
day's stock return** and compares them with a leakage-free walk-forward
backtest.

> ⚠️ **Educational only. Not financial advice. This will not make you
> rich.** Daily returns are overwhelmingly noise; near-efficient markets
> mean any real edge is tiny and fragile. Backtests systematically
> *overstate* live results (overfitting, regime change, slippage, fees,
> taxes). Do not trade real money based on this code.

## What it does

1. **Data** (`data.py`) — live prices via `yfinance`, falling back to a
   local CSV cache, then to a deterministic synthetic series so the
   pipeline always runs offline.
2. **Features** (`features.py`) — strictly causal technical features
   (lagged returns, MA ratios, volatility, RSI, momentum, volume z-score).
   Label = next-day return.
3. **Models** (`models.py`) — naive persistence baseline, Ridge,
   RandomForest, GradientBoosting, an MLP, and a PyTorch **LSTM** on
   feature sequences. All share one `fit`/`predict` interface.
4. **Backtest** (`backtest.py`) — expanding-window walk-forward: train
   only on the past, scale with train-only statistics, predict the next
   contiguous segment. A long/flat strategy with per-trade transaction
   costs is compared against buy-and-hold.

## Run

```bash
pip install -r stock_model/requirements.txt
python -m stock_model.main --ticker AAPL --period 5y --folds 5
```

Outputs:

- `stock_model/figures/equity_curves.pdf` — out-of-sample equity per model
- `stock_model/figures/model_edge.pdf` — directional accuracy vs a coin flip
- `stock_model/processed/metrics.csv` — RMSE, directional accuracy, total
  return, Sharpe, max drawdown per model

## How to read the results

- **Directional accuracy near 50%** is the expected, honest outcome on
  real data. A model that "wins" the backtest has usually overfit; only
  a robust, persistent edge across many tickers and out-of-sample years
  is even weak evidence of skill.
- `Naive(persistence)` and `BuyAndHold` are the references. If the fancy
  models cannot consistently beat both *after costs*, they add no value —
  which is the usual, expected result.

This sandbox blocks Yahoo Finance, so runs here use the synthetic
generator (clearly labelled in the output). Run it on a networked machine
for real tickers.
