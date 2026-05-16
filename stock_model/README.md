# stock_model — return prediction, significance testing, volatility forecasting

A small, honest pipeline with three leakage-free walk-forward tracks:
predict next-day return *direction* (mostly noise), test whether any
backtest edge is **real or luck**, and forecast **volatility** (where a
genuine signal exists).

> ⚠️ **Educational only. Not financial advice. This will not make you
> rich.** Daily returns are overwhelmingly noise; near-efficient markets
> mean any real direction edge is tiny and fragile. Backtests
> systematically *overstate* live results (overfitting, regime change,
> slippage, fees, taxes). Do not trade real money based on this code.

## What it does

1. **Data** (`data.py`) — live prices via `yfinance`, falling back to a
   local CSV cache, then a deterministic synthetic series so it always
   runs offline.
2. **Features** (`features.py`) — strictly causal technical features.
   Label = next-day return.
3. **Return models** (`models.py`) — naive persistence, Ridge,
   RandomForest, GradientBoosting, MLP, and a PyTorch **LSTM**.
4. **Return backtest** (`backtest.py`) — expanding-window walk-forward,
   train-only scaling, long/flat strategy with per-trade costs vs
   buy-and-hold.
5. **Significance** (`significance.py`) — the apparent backtest "winner"
   is almost always overfit. Block-bootstrap CI + one-sided p-value for
   Sharpe > 0, a paired bootstrap for "beats buy-and-hold per day", and a
   block-**permutation** test that destroys the prediction↔return
   alignment to build a proper *luck* null. Block methods preserve the
   serial correlation that i.i.d. resampling ignores.
6. **Volatility** (`volatility.py`) — return *size* clusters and is
   forecastable. Predicts forward `horizon`-day realized vol with
   NaiveVol, EWMA (RiskMetrics), HAR (Corsi-style), GradientBoosting and
   a fitted **GARCH(1,1)**. Scored with RMSE, MAE, QLIKE and R².

## Run

```bash
pip install -r stock_model/requirements.txt
python -m stock_model.main --ticker AAPL --period 5y --folds 5
```

Outputs (`stock_model/figures/`, `stock_model/processed/`):

- `equity_curves.pdf`, `metrics.csv` — return backtest
- `model_edge.pdf`, `significance.csv` — edge vs a coin flip + luck tests
- `volatility_forecast.pdf`, `volatility_metrics.csv` — vol track

## How to read the results

- **Return track:** directional accuracy near 50% is the expected, honest
  outcome. Any "winning" model is usually overfit.
- **Significance:** for the return strategies, expect `p(sharpe<=0)` and
  `perm_p(luck)` to be large (≳0.1) and the Sharpe CI to straddle zero —
  i.e. *not distinguishable from luck*. That is the point: it stops you
  fooling yourself with a lucky backtest.
- **Volatility:** expect a clearly **positive R²** and `skill_vs_naive`
  (HAR/GARCH typically beat the naive baseline on QLIKE). This real,
  measurable signal — in stark contrast to the ~0 return edge — is what
  legitimately drives risk management and position sizing. It is *not* a
  price-prediction money machine.

This sandbox blocks Yahoo Finance, so runs here use the synthetic
generator (clearly labelled in the output). It has realistic volatility
clustering, so the vol track is meaningful offline; run on a networked
machine for real tickers.
