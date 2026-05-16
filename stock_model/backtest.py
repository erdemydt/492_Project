"""Walk-forward evaluation and a cost-aware long/flat backtest.

The test span is split into `n_folds` contiguous segments. For each
segment we train on *everything before it* (expanding window), scale with
statistics fit on the training rows only, then predict the segment. Folds
are concatenated into one out-of-sample series used for every metric and
equity curve. No shuffling, no future leakage, transaction costs charged.
"""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .models import NaivePersistence

TRADING_DAYS = 252


def _max_drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    return float((equity / peak - 1.0).min())


def _sharpe(daily: np.ndarray) -> float:
    sd = daily.std()
    if sd == 0:
        return 0.0
    return float(daily.mean() / sd * np.sqrt(TRADING_DAYS))


def _strategy(signal: np.ndarray, fwd_ret: np.ndarray, cost_bps: float):
    """Long when predicted return > 0, else flat. Cost charged on turnover."""
    pos = (signal > 0).astype(float)
    turnover = np.abs(np.diff(np.concatenate([[0.0], pos])))
    daily = pos * fwd_ret - turnover * (cost_bps / 1e4)
    equity = np.cumprod(1.0 + daily)
    return daily, equity


def walk_forward(models, X, y, prices, n_folds=5, cost_bps=5.0, test_frac=0.4):
    """Return (metrics_df, equity_curves, oos_index, details).

    `details` carries the raw out-of-sample series the significance test
    resamples: per-model daily strategy returns, per-model predictions,
    the forward returns, and the buy-and-hold daily series.
    """
    X_arr = X.to_numpy(dtype=float)
    y_arr = y.to_numpy(dtype=float)
    n = len(X_arr)

    test_start = int(n * (1 - test_frac))
    bounds = np.linspace(test_start, n, n_folds + 1, dtype=int)

    oos_pred = {name: np.full(n - test_start, np.nan) for name in models}
    oos_slice = slice(test_start, n)
    fwd_ret = y_arr[oos_slice]

    for k in range(n_folds):
        lo, hi = bounds[k], bounds[k + 1]
        tr = slice(0, lo)

        scaler = StandardScaler().fit(X_arr[tr])
        Xtr, Xte = scaler.transform(X_arr[tr]), scaler.transform(X_arr[lo:hi])
        ytr = y_arr[tr]

        for name, proto in models.items():
            model = copy.deepcopy(proto)
            # The naive baseline must see the *raw* return, not the
            # standardised feature (mean-centring can flip the sign).
            if isinstance(proto, NaivePersistence):
                model.fit(X_arr[tr], ytr)
                pred = np.asarray(model.predict(X_arr[lo:hi]), dtype=float)
            else:
                model.fit(Xtr, ytr)
                pred = np.asarray(model.predict(Xte), dtype=float)
            oos_pred[name][lo - test_start : hi - test_start] = pred

    rows, curves = [], {}
    bh_daily = fwd_ret
    bh_equity = np.cumprod(1.0 + bh_daily)
    curves["BuyAndHold"] = bh_equity
    rows.append(
        {
            "model": "BuyAndHold",
            "rmse": np.nan,
            "dir_acc": float((fwd_ret > 0).mean()),
            "total_return": float(bh_equity[-1] - 1.0),
            "sharpe": _sharpe(bh_daily),
            "max_drawdown": _max_drawdown(bh_equity),
        }
    )

    daily_returns = {"BuyAndHold": bh_daily}
    for name, pred in oos_pred.items():
        rmse = float(np.sqrt(np.mean((pred - fwd_ret) ** 2)))
        dir_acc = float((np.sign(pred) == np.sign(fwd_ret)).mean())
        daily, equity = _strategy(pred, fwd_ret, cost_bps)
        curves[name] = equity
        daily_returns[name] = daily
        rows.append(
            {
                "model": name,
                "rmse": rmse,
                "dir_acc": dir_acc,
                "total_return": float(equity[-1] - 1.0),
                "sharpe": _sharpe(daily),
                "max_drawdown": _max_drawdown(equity),
            }
        )

    metrics = pd.DataFrame(rows).set_index("model")
    details = {
        "daily_returns": daily_returns,
        "predictions": oos_pred,
        "fwd_ret": fwd_ret,
    }
    return metrics, curves, prices.index[oos_slice], details
