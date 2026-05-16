"""Price data loading.

Resolution order:
  1. yfinance (live Yahoo Finance) — works on a networked machine.
  2. Local CSV cache in stock_model/cache/ — populated by a previous live run.
  3. Synthetic generator — a deterministic, vaguely-realistic price series so
     the whole pipeline runs offline. NOT real market data; for demo only.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")


def _cache_path(ticker: str, period: str) -> str:
    return os.path.join(CACHE_DIR, f"{ticker.upper()}_{period}.csv")


def _from_yfinance(ticker: str, period: str) -> pd.DataFrame | None:
    try:
        import yfinance as yf
    except ImportError:
        return None
    try:
        df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
    except Exception:
        return None
    if df is None or len(df) == 0:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    return df


def _from_cache(ticker: str, period: str) -> pd.DataFrame | None:
    path = _cache_path(ticker, period)
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df if len(df) else None


def _synthetic(period: str, seed: int = 42) -> pd.DataFrame:
    """Geometric Brownian motion with mild momentum and volatility
    clustering, so models have a faint (but weak) signal to find — much
    like a real, near-efficient market.
    """
    n_days = {"1y": 252, "2y": 504, "5y": 1260, "10y": 2520, "max": 2520}.get(period, 1260)
    rng = np.random.default_rng(seed)

    mu = 0.0003          # ~7.8%/yr drift
    log_vol = np.log(0.012)
    rets = np.empty(n_days)
    prev_ret = 0.0
    for t in range(n_days):
        log_vol = 0.97 * log_vol + 0.03 * np.log(0.012) + rng.normal(0, 0.15)
        vol = np.exp(log_vol)
        shock = rng.normal(0, vol)
        rets[t] = mu + 0.05 * prev_ret + shock  # weak autocorrelation
        prev_ret = rets[t]

    close = 100.0 * np.exp(np.cumsum(rets))
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n_days)
    intraday = np.abs(rng.normal(0, 0.005, n_days))
    df = pd.DataFrame(
        {
            "Open": close * (1 - rng.normal(0, 0.003, n_days)),
            "High": close * (1 + intraday),
            "Low": close * (1 - intraday),
            "Close": close,
            "Volume": rng.integers(1_000_000, 5_000_000, n_days),
        },
        index=idx,
    )
    return df


def get_prices(ticker: str = "AAPL", period: str = "5y", allow_synthetic: bool = True):
    """Return (DataFrame[OHLCV], source_str)."""
    df = _from_yfinance(ticker, period)
    if df is not None:
        os.makedirs(CACHE_DIR, exist_ok=True)
        df.to_csv(_cache_path(ticker, period))
        return df, f"yfinance:{ticker}"

    df = _from_cache(ticker, period)
    if df is not None:
        return df, f"cache:{ticker}"

    if allow_synthetic:
        return _synthetic(period), "synthetic"

    raise RuntimeError(
        f"No data for {ticker}: Yahoo Finance unreachable and no cache. "
        "Run on a networked machine or supply a cached CSV."
    )
