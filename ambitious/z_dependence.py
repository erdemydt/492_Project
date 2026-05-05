"""z_dependence.py — Fit δ(B) = A * B^n per z-layer and compute α(z).

Reads: ambitious/processed/falloff_2d_by_z.csv
Writes: ambitious/processed/z_scaling_by_z.csv, ambitious/figures/alpha_vs_z.pdf,
        ambitious/figures/exponent_vs_z.pdf

Produces robust fits only for z-layers with sufficient valid points.
"""

from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


ROOT = Path(__file__).resolve().parent
PROCESSED = ROOT / "processed"
FIGS = ROOT / "figures"
PROCESSED.mkdir(exist_ok=True)
FIGS.mkdir(exist_ok=True)

R_IN_MM = 0.1
H_M = 1e-3
MU = 1.95e-3
SIGMA = 3.7e6


def D_m(B0):
    return np.sqrt(MU / (SIGMA * np.asarray(B0) ** 2))


def power_law(B, A, n):
    return A * B ** n


def robust_fit(B, delta, p0=(2e-4, -0.5)):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        popt, pcov = curve_fit(power_law, B, delta, p0=p0, maxfev=10000)
    perr = np.sqrt(np.diag(pcov)) if pcov is not None else (np.nan, np.nan)
    return popt, perr


def main():
    df = pd.read_csv(PROCESSED / "falloff_2d_by_z.csv")
    df = df.dropna(subset=["r_half_mm"])  # remove rows where r_half not found

    # The raw export has B-dependent tiny z offsets; bin to a common z grid
    df["z_q"] = df["z_mm"].round(3)  # mm rounding to 0.001mm (~1 µm)
    z_values = sorted(df["z_q"].unique())
    rows = []

    # We'll fit only layers with at least 6 valid points
    min_points = 6
    for z in z_values:
        sub = df[df["z_q"] == z].copy()
        sub = sub.dropna(subset=["delta_half_um"])  # µm
        if len(sub) < min_points:
            continue

        B = sub["B0_T"].to_numpy(dtype=float)
        delta_m = sub["delta_half_um"].to_numpy(dtype=float) * 1e-6

        try:
            (A_fit, n_fit), (A_err, n_err) = robust_fit(B, delta_m)
        except Exception:
            A_fit = np.nan; n_fit = np.nan; A_err = np.nan; n_err = np.nan

        # compute alpha statistics for layer
        D_layer = D_m(B)
        sqrtDh = np.sqrt(D_layer * H_M)
        alpha_vals = np.divide(delta_m, sqrtDh, out=np.full_like(delta_m, np.nan), where=(sqrtDh>0))
        alpha_med = float(np.nanmedian(alpha_vals))
        alpha_mean = float(np.nanmean(alpha_vals))
        alpha_std = float(np.nanstd(alpha_vals))

        rows.append({
            "z_mm": float(np.median(sub["z_mm"])),
            "n_points": int(len(sub)),
            "A_fit": float(A_fit) if np.isfinite(A_fit) else np.nan,
            "A_err": float(A_err) if np.isfinite(A_err) else np.nan,
            "n_fit": float(n_fit) if np.isfinite(n_fit) else np.nan,
            "n_err": float(n_err) if np.isfinite(n_err) else np.nan,
            "alpha_med": alpha_med,
            "alpha_mean": alpha_mean,
            "alpha_std": alpha_std,
        })

    res = pd.DataFrame(rows).sort_values("z_mm").reset_index(drop=True)
    res.to_csv(PROCESSED / "z_scaling_by_z.csv", index=False)

    if res.empty:
        print("No z-layers had enough points for per-layer fits.")
        return

    # Plot alpha (median) vs z
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(res["z_mm"], res["alpha_med"], "o-", color="#2b8cbe")
    ax.fill_between(res["z_mm"], res["alpha_med"] - res["alpha_std"], res["alpha_med"] + res["alpha_std"],
                    color="#2b8cbe", alpha=0.2)
    ax.axvline(0.5, color="k", ls="--", lw=0.8)
    ax.set_xlabel("z (mm)")
    ax.set_ylabel(r"median $\,\alpha = \delta/\sqrt{Dh}$")
    ax.set_title("Alpha (median) vs z")
    fig.tight_layout()
    fig.savefig(FIGS / "alpha_vs_z.pdf")
    plt.close(fig)

    # Plot exponent n vs z
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(res["z_mm"], res["n_fit"], "o-", color="#7b3294")
    ax.fill_between(res["z_mm"], res["n_fit"] - res["n_err"], res["n_fit"] + res["n_err"],
                    color="#7b3294", alpha=0.2)
    ax.axhline(-0.5, color="k", ls="--", lw=0.8)
    ax.axvline(0.5, color="k", ls="--", lw=0.8)
    ax.set_xlabel("z (mm)")
    ax.set_ylabel(r"fitted exponent $n$ in $\delta = A\,B^n$")
    ax.set_title("Fitted exponent n vs z")
    fig.tight_layout()
    fig.savefig(FIGS / "exponent_vs_z.pdf")
    plt.close(fig)

    print(f"Saved {len(res)} per-z fits to {PROCESSED / 'z_scaling_by_z.csv'}")
    print(f"Saved figures to {FIGS / 'alpha_vs_z.pdf'} and {FIGS / 'exponent_vs_z.pdf'}")


if __name__ == "__main__":
    main()
