"""Causal feature engineering.

Every feature for day t uses only information available at the close of
day t. The label is the *next* day's simple return (t -> t+1). This strict
causality is what keeps the backtest honest — no lookahead leakage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLS = [
    "ret_1",
    "ret_5",
    "ret_10",
    "ma_ratio_5",
    "ma_ratio_10",
    "ma_ratio_20",
    "vol_10",
    "vol_20",
    "rsi_14",
    "mom_10",
    "range_pct",
    "volume_z",
]


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def build_dataset(df: pd.DataFrame):
    """Return (X DataFrame, y Series, prices Series) aligned by date."""
    close = df["Close"].astype(float)
    ret = close.pct_change()

    feat = pd.DataFrame(index=df.index)
    feat["ret_1"] = ret
    feat["ret_5"] = close.pct_change(5)
    feat["ret_10"] = close.pct_change(10)
    feat["ma_ratio_5"] = close / close.rolling(5).mean() - 1
    feat["ma_ratio_10"] = close / close.rolling(10).mean() - 1
    feat["ma_ratio_20"] = close / close.rolling(20).mean() - 1
    feat["vol_10"] = ret.rolling(10).std()
    feat["vol_20"] = ret.rolling(20).std()
    feat["rsi_14"] = _rsi(close, 14) / 100.0
    feat["mom_10"] = close / close.shift(10) - 1
    feat["range_pct"] = (df["High"] - df["Low"]) / close
    vol = df["Volume"].astype(float)
    feat["volume_z"] = (vol - vol.rolling(20).mean()) / vol.rolling(20).std()

    # Label: next-day return. shift(-1) so row t holds the t -> t+1 move.
    y = ret.shift(-1)

    data = feat.join(y.rename("target")).dropna()
    X = data[FEATURE_COLS]
    y = data["target"]
    prices = close.loc[X.index]
    return X, y, prices
