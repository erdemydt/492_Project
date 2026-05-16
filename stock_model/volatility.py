"""Volatility-forecasting track.

Unlike the *direction* of returns, the *size* of returns clusters and is
genuinely forecastable. Here we predict forward `horizon`-day realized
volatility and compare:

  * NaiveVol     - trailing realized vol (persistence baseline)
  * EWMAVol      - RiskMetrics EWMA (lambda = 0.94), parameter-free
  * HAR          - multi-horizon OLS on trailing 5/10/22-day vol (Corsi-style)
  * GradientBoosting - on the full causal vol feature set
  * GARCH(1,1)   - fitted per fold via `arch`, analytic h-day-ahead forecast

Same expanding walk-forward as the return backtest: train only on the
past, predict the next contiguous segment, concatenate to one
out-of-sample series. Scored with RMSE, MAE, QLIKE (the standard robust
volatility loss) and R^2 — which, for volatility, is meaningfully
positive. That contrast with the ~0 R^2 of return prediction is the whole
point of this track.
"""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression

HAR_COLS = ["rv5", "rv10", "rv22"]
GBR_COLS = ["rv5", "rv10", "rv22", "ewma", "abs_r", "parkinson"]


def build_vol_dataset(df: pd.DataFrame, horizon: int = 5):
    """Return (D DataFrame, target Series, returns Series), date-aligned.

    Target = realized vol of the *next* `horizon` daily returns. Every
    feature uses only data up to and including day t.
    """
    close = df["Close"].astype(float)
    r = close.pct_change()

    D = pd.DataFrame(index=df.index)
    D["r"] = r
    D["abs_r"] = r.abs()
    D["rv5"] = r.rolling(5).std()
    D["rv10"] = r.rolling(10).std()
    D["rv22"] = r.rolling(22).std()
    D["rv_h"] = r.rolling(horizon).std()  # trailing-horizon vol = naive forecast
    D["ewma"] = r.ewm(alpha=1 - 0.94).std()
    with np.errstate(invalid="ignore"):
        park = np.sqrt((np.log(df["High"] / df["Low"]) ** 2) / (4 * np.log(2)))
    D["parkinson"] = park

    # Forward realized vol: std of r over the next `horizon` days.
    fwd = r.shift(-1).rolling(horizon).std().shift(-(horizon - 1))
    D["target"] = fwd

    D = D.dropna()
    return D, D["target"], D["r"]


def _garch_forecast(train_r, full_r, lo, hi, horizon):
    """Per-fold GARCH(1,1): fit on train returns, then run the conditional
    variance recursion forward and average the analytic multi-step forecast
    over the horizon for each test day. Returns vol (std) predictions.
    """
    from arch import arch_model

    s = 100.0  # arch is happiest with returns in percent
    res = arch_model(train_r.to_numpy() * s, mean="Constant", vol="GARCH",
                     p=1, q=1).fit(disp="off")
    pr = res.params
    mu = pr["mu"]
    omega, alpha, beta = pr["omega"], pr["alpha[1]"], pr["beta[1]"]
    phi = alpha + beta
    uncond = omega / (1 - phi) if phi < 1 else omega

    rs = full_r.to_numpy() * s
    sig2 = np.empty(len(rs))
    sig2[0] = uncond
    for t in range(1, len(rs)):
        eps = rs[t - 1] - mu
        sig2[t] = omega + alpha * eps * eps + beta * sig2[t - 1]

    out = np.empty(hi - lo)
    ks = np.arange(horizon)
    for p in range(lo, hi):
        eps = rs[p] - mu
        var1 = omega + alpha * eps * eps + beta * sig2[p]  # 1-step ahead
        multi = uncond + (phi ** ks) * (var1 - uncond)     # k-step path
        out[p - lo] = np.sqrt(multi.mean()) / s
    return out


def _metrics(name, pred, true, naive_rmse=None):
    err = pred - true
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    f2 = np.clip(pred, 1e-8, None) ** 2
    rv2 = np.clip(true, 1e-8, None) ** 2
    qlike = float(np.mean(rv2 / f2 - np.log(rv2 / f2) - 1.0))
    ss_res = np.sum(err ** 2)
    ss_tot = np.sum((true - true.mean()) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    skill = float(1 - rmse / naive_rmse) if naive_rmse else np.nan
    return {"model": name, "rmse": rmse, "mae": mae, "qlike": qlike,
            "r2": r2, "skill_vs_naive": skill}


def walk_forward_vol(D, target, returns, horizon=5, n_folds=5, test_frac=0.4):
    """Return (metrics_df, predictions dict, oos_index)."""
    n = len(D)
    test_start = int(n * (1 - test_frac))
    bounds = np.linspace(test_start, n, n_folds + 1, dtype=int)
    oos = slice(test_start, n)
    true = target.to_numpy()[oos]

    names = ["NaiveVol", "EWMAVol", "HAR", "GradientBoosting"]
    try:
        import arch  # noqa: F401

        names.append("GARCH(1,1)")
    except ImportError:
        pass
    preds = {nm: np.full(n - test_start, np.nan) for nm in names}

    Xhar = D[HAR_COLS].to_numpy(float)
    Xgbr = D[GBR_COLS].to_numpy(float)
    y = target.to_numpy(float)

    for k in range(n_folds):
        lo, hi = bounds[k], bounds[k + 1]
        sl = slice(lo - test_start, hi - test_start)

        preds["NaiveVol"][sl] = D["rv_h"].to_numpy()[lo:hi]
        preds["EWMAVol"][sl] = D["ewma"].to_numpy()[lo:hi]

        har = LinearRegression().fit(Xhar[:lo], y[:lo])
        preds["HAR"][sl] = np.clip(har.predict(Xhar[lo:hi]), 1e-8, None)

        gbr = GradientBoostingRegressor(
            n_estimators=150, max_depth=2, learning_rate=0.03,
            subsample=0.8, random_state=0
        ).fit(Xgbr[:lo], y[:lo])
        preds["GradientBoosting"][sl] = np.clip(gbr.predict(Xgbr[lo:hi]), 1e-8, None)

        if "GARCH(1,1)" in preds:
            try:
                preds["GARCH(1,1)"][sl] = _garch_forecast(
                    returns.iloc[:lo], returns, lo, hi, horizon
                )
            except Exception:
                preds["GARCH(1,1)"][sl] = D["rv_h"].to_numpy()[lo:hi]

    naive_rmse = float(np.sqrt(np.mean((preds["NaiveVol"] - true) ** 2)))
    rows = [_metrics(nm, preds[nm], true, naive_rmse) for nm in names]
    metrics = pd.DataFrame(rows).set_index("model")
    return metrics, preds, D.index[oos]
