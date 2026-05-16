"""Is the backtest edge real, or luck?

Daily PnL is autocorrelated, so naive i.i.d. resampling understates the
noise. These tests use *block* methods that preserve short-range serial
structure:

  * block-bootstrap CI + one-sided p-value for Sharpe > 0,
  * paired block bootstrap for "strategy beats buy-and-hold per day",
  * block-permutation test that destroys the prediction<->return
    alignment while keeping each series' own autocorrelation, giving a
    proper "luck" null for the directional edge.

A strategy that cannot clear these bars is indistinguishable from chance,
which is the usual, expected outcome for daily return prediction.
"""

from __future__ import annotations

import numpy as np

from .backtest import TRADING_DAYS, _sharpe, _strategy


def _moving_block_resample(x: np.ndarray, block: int, rng: np.random.Generator) -> np.ndarray:
    """Length-preserving moving-block bootstrap sample."""
    n = len(x)
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=n_blocks)
    out = np.concatenate([x[s : s + block] for s in starts])
    return out[:n]


def _block_permute(x: np.ndarray, block: int, rng: np.random.Generator) -> np.ndarray:
    """Shuffle the order of contiguous blocks (keeps within-block structure)."""
    n = len(x)
    blocks = [x[i : i + block] for i in range(0, n, block)]
    order = rng.permutation(len(blocks))
    return np.concatenate([blocks[i] for i in order])[:n]


def bootstrap_sharpe(daily: np.ndarray, n_boot=2000, block=10, seed=0) -> dict:
    """One-sided test of H0: Sharpe <= 0, with a 95% CI."""
    rng = np.random.default_rng(seed)
    obs = _sharpe(daily)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        s = _moving_block_resample(daily, block, rng)
        boot[b] = s.mean() / s.std() * np.sqrt(TRADING_DAYS) if s.std() > 0 else 0.0
    return {
        "sharpe": obs,
        "ci_low": float(np.percentile(boot, 2.5)),
        "ci_high": float(np.percentile(boot, 97.5)),
        "p_value": float((boot <= 0).mean()),  # P(no positive Sharpe under resampling)
    }


def paired_vs_benchmark(strat: np.ndarray, bench: np.ndarray, n_boot=2000, block=10, seed=0) -> dict:
    """One-sided test of H0: mean(strategy - benchmark) <= 0 per day."""
    rng = np.random.default_rng(seed)
    diff = strat - bench
    obs = float(diff.mean())
    boot = np.array(
        [_moving_block_resample(diff, block, rng).mean() for _ in range(n_boot)]
    )
    return {
        "mean_daily_excess": obs,
        "ci_low": float(np.percentile(boot, 2.5)),
        "ci_high": float(np.percentile(boot, 97.5)),
        "p_value": float((boot <= 0).mean()),
    }


def permutation_edge(pred, fwd_ret, cost_bps=5.0, n_perm=2000, block=10, seed=0) -> dict:
    """Null = predictions hold no information about future returns.

    Block-permute the predictions (destroying alignment with returns but
    preserving the predictions' own serial structure) and recompute the
    strategy Sharpe. p = fraction of the null at least as good as observed.
    """
    rng = np.random.default_rng(seed)
    obs_daily, _ = _strategy(pred, fwd_ret, cost_bps)
    obs = _sharpe(obs_daily)
    null = np.empty(n_perm)
    for i in range(n_perm):
        perm = _block_permute(np.asarray(pred, dtype=float), block, rng)
        d, _ = _strategy(perm, fwd_ret, cost_bps)
        null[i] = _sharpe(d)
    p = (1 + int((null >= obs).sum())) / (1 + n_perm)
    return {
        "sharpe": obs,
        "null_mean": float(null.mean()),
        "null_p95": float(np.percentile(null, 95)),
        "p_value": float(p),
    }


def run_all(details: dict, cost_bps=5.0, n_boot=2000, seed=0) -> "object":
    """Build a tidy results table over every traded model in `details`."""
    import pandas as pd

    bench = details["daily_returns"]["BuyAndHold"]
    rows = []
    for name, daily in details["daily_returns"].items():
        if name == "BuyAndHold":
            continue
        sh = bootstrap_sharpe(daily, n_boot=n_boot, seed=seed)
        pv = paired_vs_benchmark(daily, bench, n_boot=n_boot, seed=seed)
        pm = permutation_edge(details["predictions"][name], details["fwd_ret"],
                              cost_bps=cost_bps, n_perm=n_boot, seed=seed)
        rows.append(
            {
                "model": name,
                "sharpe": sh["sharpe"],
                "sharpe_ci": f"[{sh['ci_low']:.2f}, {sh['ci_high']:.2f}]",
                "p(sharpe<=0)": sh["p_value"],
                "daily_excess_vs_BH": pv["mean_daily_excess"],
                "p(<=BH)": pv["p_value"],
                "perm_p(luck)": pm["p_value"],
            }
        )
    return pd.DataFrame(rows).set_index("model")
