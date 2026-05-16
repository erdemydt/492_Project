"""Stock-modelling demo — entry point.

  python -m stock_model.main --ticker AAPL --period 5y --folds 5

Three things, all walk-forward and leakage-free:
  1. Return-direction backtest across several models (mostly noise — the
     honest, expected result).
  2. A significance test: block-bootstrap + permutation, asking whether
     any backtest edge is distinguishable from luck.
  3. A volatility-forecasting track, where there *is* a real, measurable
     signal.

Educational only — read the disclaimer below.
"""

from __future__ import annotations

import argparse
import os

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .backtest import walk_forward
from .data import get_prices
from .features import build_dataset
from .models import build_models
from .significance import run_all as run_significance
from .volatility import build_vol_dataset, walk_forward_vol

HERE = os.path.dirname(__file__)
FIG_DIR = os.path.join(HERE, "figures")
PROC_DIR = os.path.join(HERE, "processed")

DISCLAIMER = """
================================================================================
  EDUCATIONAL DEMO — NOT FINANCIAL ADVICE
  Daily stock returns are dominated by noise. These models will not reliably
  predict the market or make you rich. Backtested results overstate live
  performance (overfitting, regime change, slippage, taxes). Do not trade
  real money on this. Past performance does not predict future returns.
================================================================================
"""


def _fmt(df):
    with pd.option_context("display.float_format", "{:.4f}".format):
        return df.to_string()


def _plot_equity(curves, index, ticker, source, outpath):
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, eq in curves.items():
        style = "--" if name == "BuyAndHold" else "-"
        lw = 2.0 if name == "BuyAndHold" else 1.3
        ax.plot(index, eq, style, lw=lw, label=name)
    ax.set_title(f"Out-of-sample equity — {ticker} ({source})")
    ax.set_xlabel("Date")
    ax.set_ylabel("Growth of 1 unit")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)


def _plot_edge(metrics, outpath):
    fig, ax = plt.subplots(figsize=(7, 4))
    m = metrics.drop(index="BuyAndHold", errors="ignore")
    ax.bar(m.index, (m["dir_acc"] - 0.5) * 100, color="steelblue")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("Directional accuracy edge over 50%  (pp)")
    ax.set_title("Return prediction: edge over a coin flip (≈0 expected)")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)


def _plot_vol(vmetrics, vpreds, target, vidx, ticker, outpath):
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(11, 4))

    a0.plot(vidx, target, color="0.4", lw=1.0, label="Realized")
    for nm in ("NaiveVol", "HAR", "GARCH(1,1)"):
        if nm in vpreds:
            a0.plot(vidx, vpreds[nm], lw=1.0, label=nm)
    a0.set_title(f"Forward realized volatility — {ticker}")
    a0.set_xlabel("Date")
    a0.set_ylabel("Vol (daily-return std)")
    a0.legend(fontsize=7)
    a0.grid(alpha=0.3)

    a1.bar(vmetrics.index, vmetrics["r2"], color="seagreen")
    a1.axhline(0, color="k", lw=0.8)
    a1.set_title("Volatility R²  (positive = real, predictable signal)")
    a1.set_ylabel("Out-of-sample R²")
    a1.tick_params(axis="x", rotation=30)
    a1.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="Stock modelling demo")
    p.add_argument("--ticker", default="AAPL")
    p.add_argument("--period", default="5y", help="1y, 2y, 5y, 10y, max")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seq-len", type=int, default=20, help="LSTM lookback window")
    p.add_argument("--cost-bps", type=float, default=5.0, help="per-trade cost (bps)")
    p.add_argument("--horizon", type=int, default=5, help="vol forecast horizon (days)")
    p.add_argument("--n-boot", type=int, default=2000, help="bootstrap/permutation reps")
    args = p.parse_args()

    print(DISCLAIMER)

    df, source = get_prices(args.ticker, args.period)
    if source == "synthetic":
        print("[!] Live data unavailable — using SYNTHETIC prices (demo only).")
    print(f"[i] Data source: {source}   rows: {len(df)}")

    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(PROC_DIR, exist_ok=True)

    # ---- 1. Return-direction backtest -------------------------------------
    X, y, prices = build_dataset(df)
    models = build_models(seq_len=args.seq_len)
    print(f"[i] Return models: {', '.join(models)}")
    metrics, curves, oos_index, details = walk_forward(
        models, X, y, prices, n_folds=args.folds, cost_bps=args.cost_bps
    )
    print(f"\n=== 1. Return backtest ({len(oos_index)} OOS days, "
          f"{args.cost_bps} bps/trade) ===")
    print(_fmt(metrics))

    # ---- 2. Significance test ---------------------------------------------
    sig = run_significance(details, cost_bps=args.cost_bps, n_boot=args.n_boot)
    print("\n=== 2. Is the edge real? (block bootstrap + permutation) ===")
    print(_fmt(sig))
    print("    p(sharpe<=0): chance the positive Sharpe is a resampling fluke.")
    print("    perm_p(luck): chance random predictions match this Sharpe.")
    print("    Values near/above ~0.10 mean: not distinguishable from luck.")

    # ---- 3. Volatility-forecasting track ----------------------------------
    D, vtarget, vret = build_vol_dataset(df, horizon=args.horizon)
    vmetrics, vpreds, vidx = walk_forward_vol(
        D, vtarget, vret, horizon=args.horizon, n_folds=args.folds
    )
    print(f"\n=== 3. Volatility forecast ({len(vidx)} OOS days, "
          f"horizon={args.horizon}d) ===")
    print(_fmt(vmetrics))
    print("    Positive R² and skill_vs_naive => a genuine, usable signal,")
    print("    in stark contrast to the ~0 edge of return prediction above.")

    # ---- Outputs ----------------------------------------------------------
    eq_path = os.path.join(FIG_DIR, "equity_curves.pdf")
    edge_path = os.path.join(FIG_DIR, "model_edge.pdf")
    vol_path = os.path.join(FIG_DIR, "volatility_forecast.pdf")
    _plot_equity(curves, oos_index, args.ticker, source, eq_path)
    _plot_edge(metrics, edge_path)
    _plot_vol(vmetrics, vpreds, vtarget.to_numpy()[-len(vidx):], vidx,
              args.ticker, vol_path)

    metrics.to_csv(os.path.join(PROC_DIR, "metrics.csv"))
    sig.to_csv(os.path.join(PROC_DIR, "significance.csv"))
    vmetrics.to_csv(os.path.join(PROC_DIR, "volatility_metrics.csv"))

    print(f"\n[i] Figures -> {FIG_DIR}")
    print(f"[i] CSVs    -> {PROC_DIR}")
    print(DISCLAIMER)


if __name__ == "__main__":
    main()
