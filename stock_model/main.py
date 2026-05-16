"""Multi-model stock-prediction comparison — entry point.

  python -m stock_model.main --ticker AAPL --period 5y --folds 5

Trains several models to predict the next day's return, runs an honest
walk-forward backtest with transaction costs, prints a metrics table and
writes figures + a CSV. Educational only — read the disclaimer below.
"""

from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .backtest import walk_forward
from .data import get_prices
from .features import build_dataset
from .models import build_models

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


def _plot_metrics(metrics, outpath):
    fig, ax = plt.subplots(figsize=(7, 4))
    m = metrics.drop(index="BuyAndHold", errors="ignore")
    ax.bar(m.index, (m["dir_acc"] - 0.5) * 100, color="steelblue")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("Directional accuracy edge over 50%  (pp)")
    ax.set_title("Predictive edge by model (above 0 = better than a coin flip)")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="Multi-model stock prediction comparison")
    p.add_argument("--ticker", default="AAPL")
    p.add_argument("--period", default="5y", help="1y, 2y, 5y, 10y, max")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seq-len", type=int, default=20, help="LSTM lookback window")
    p.add_argument("--cost-bps", type=float, default=5.0, help="per-trade cost (bps)")
    args = p.parse_args()

    print(DISCLAIMER)

    df, source = get_prices(args.ticker, args.period)
    if source == "synthetic":
        print("[!] Live data unavailable — using SYNTHETIC prices (demo only).")
    print(f"[i] Data source: {source}   rows: {len(df)}")

    X, y, prices = build_dataset(df)
    print(f"[i] Samples: {len(X)}   Features: {X.shape[1]}")

    models = build_models(seq_len=args.seq_len)
    print(f"[i] Models: {', '.join(models)}")

    metrics, curves, oos_index = walk_forward(
        models, X, y, prices, n_folds=args.folds, cost_bps=args.cost_bps
    )

    print("\n=== Walk-forward out-of-sample results "
          f"({len(oos_index)} days, {args.cost_bps} bps/trade) ===")
    with __import__("pandas").option_context("display.float_format", "{:.4f}".format):
        print(metrics.to_string())

    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(PROC_DIR, exist_ok=True)
    eq_path = os.path.join(FIG_DIR, "equity_curves.pdf")
    edge_path = os.path.join(FIG_DIR, "model_edge.pdf")
    csv_path = os.path.join(PROC_DIR, "metrics.csv")
    _plot_equity(curves, oos_index, args.ticker, source, eq_path)
    _plot_metrics(metrics, edge_path)
    metrics.to_csv(csv_path)

    print(f"\n[i] Wrote {eq_path}")
    print(f"[i] Wrote {edge_path}")
    print(f"[i] Wrote {csv_path}")
    print(DISCLAIMER)


if __name__ == "__main__":
    main()
