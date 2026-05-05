"""z_dependence_2d.py — Wall-to-midplane analysis for the 2D B-sweep.

This script reads processed/falloff_2d_by_z.csv, uses the symmetry about the
midplane to rewrite the data as a nearest-wall coordinate eta = min(z, 1-z),
and quantifies how the fall-off law changes from the Hartmann-wall region to
the bulk midplane.

Outputs:
  figures/z_dependence_2d.pdf
  processed/z_dependence_2d_metrics.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.optimize import curve_fit


ROOT = Path(__file__).resolve().parent
PROCESSED_DIR = ROOT / "processed"
FIGURES_DIR = ROOT / "figures"

SIGMA = 3.7e6
MU = 1.95e-3
H_M = 1.0e-3


def hartmann_thickness_m(b0: np.ndarray | float) -> np.ndarray | float:
    return np.sqrt(MU / (SIGMA * np.asarray(b0) ** 2))


def power_law(x: np.ndarray, a: float, n: float) -> np.ndarray:
    return a * x**n


def build_half_profile(sub: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    half = sub.copy()
    half["eta"] = np.minimum(half["z_mm"].to_numpy(float), 1.0 - half["z_mm"].to_numpy(float))
    half = half.sort_values("eta")
    half = half.dropna(subset=["delta_half_um"])
    half = half[half["eta"] <= 0.5]
    return half["eta"].to_numpy(float), half["delta_half_um"].to_numpy(float)


def interpolate_band(df: pd.DataFrame, eta_target: float) -> tuple[np.ndarray, np.ndarray]:
    bvals: list[float] = []
    deltas: list[float] = []
    for b0, sub in df.groupby("B0_T"):
        eta, delta = build_half_profile(sub)
        if len(delta) < 4:
            continue
        if eta_target < eta.min() or eta_target > eta.max():
            continue
        value = np.interp(eta_target, eta, delta)
        if np.isfinite(value):
            bvals.append(float(b0))
            deltas.append(float(value) * 1e-6)
    return np.asarray(bvals, dtype=float), np.asarray(deltas, dtype=float)


def main() -> None:
    FIGURES_DIR.mkdir(exist_ok=True)

    df = pd.read_csv(PROCESSED_DIR / "falloff_2d_by_z.csv")
    df = df.sort_values(["B0_T", "z_mm"]).reset_index(drop=True)

    eta_targets = np.array([0.05, 0.10, 0.25, 0.45])
    colors = plt.cm.viridis(np.linspace(0.15, 0.9, len(eta_targets)))

    records = []
    for eta_target in eta_targets:
        bvals, deltas = interpolate_band(df, eta_target)
        if len(bvals) < 5:
            records.append(
                {
                    "eta_frac": eta_target,
                    "count": len(bvals),
                    "A_fit_um": np.nan,
                    "n_fit": np.nan,
                    "n_err": np.nan,
                    "alpha_mean": np.nan,
                    "alpha_median": np.nan,
                    "alpha_std": np.nan,
                }
            )
            continue

        popt, pcov = curve_fit(power_law, bvals, deltas, p0=[2e-4, -0.5], maxfev=10000)
        a_fit, n_fit = popt
        n_err = float(np.sqrt(np.diag(pcov))[1])

        D = hartmann_thickness_m(bvals)
        alpha = deltas / np.sqrt(D * H_M)
        records.append(
            {
                "eta_frac": eta_target,
                "count": len(bvals),
                "A_fit_um": a_fit * 1e6,
                "n_fit": n_fit,
                "n_err": n_err,
                "alpha_mean": float(np.mean(alpha)),
                "alpha_median": float(np.median(alpha)),
                "alpha_std": float(np.std(alpha)),
            }
        )

    metrics = pd.DataFrame(records).sort_values("eta_frac")
    metrics.to_csv(PROCESSED_DIR / "z_dependence_2d_metrics.csv", index=False)

    print("Wall-to-midplane 2D summary")
    print("=" * 34)
    print(metrics.to_string(index=False))

    fig, (ax_fit, ax_summary) = plt.subplots(1, 2, figsize=(13, 4.8))

    for eta_target, color in zip(eta_targets, colors):
        bvals, deltas = interpolate_band(df, eta_target)
        if len(bvals) < 5:
            continue
        popt, _ = curve_fit(power_law, bvals, deltas, p0=[2e-4, -0.5], maxfev=10000)
        a_fit, n_fit = popt
        b_grid = np.logspace(np.log10(bvals.min()), np.log10(bvals.max()), 300)
        label = rf"$\eta = {eta_target:.2f}$, $n={n_fit:.2f}$"
        ax_fit.loglog(bvals, deltas * 1e6, "o", ms=4, color=color, alpha=0.85, label=label)
        ax_fit.loglog(b_grid, power_law(b_grid, a_fit, n_fit) * 1e6, color=color, lw=1.4)

    ax_fit.set_xlabel(r"$B_0$  (T)")
    ax_fit.set_ylabel(r"$\delta_{1/2}$  (µm)")
    ax_fit.set_title("Fall-off law by distance from nearest wall")
    ax_fit.grid(True, which="both", ls=":", alpha=0.35)
    ax_fit.legend(fontsize=8, loc="lower left")
    ax_fit.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax_fit.yaxis.set_major_formatter(ticker.ScalarFormatter())

    ax_alpha = ax_summary.twinx()
    ax_summary.plot(metrics["eta_frac"], metrics["alpha_median"], "o-", color="#d95f0e", lw=1.6, label=r"median $\alpha$")
    ax_summary.plot(metrics["eta_frac"], metrics["alpha_mean"], "s--", color="#1b9e77", lw=1.2, label=r"mean $\alpha$")
    ax_alpha.errorbar(metrics["eta_frac"], metrics["n_fit"], yerr=metrics["n_err"], fmt="d-", color="#225ea8", lw=1.4, ms=5, label=r"$n$ fit")

    ax_summary.set_xlabel(r"Nearest-wall coordinate $\eta = \min(z, 1-z)$")
    ax_summary.set_ylabel(r"$\alpha = \delta_{1/2}/\sqrt{Dh}$")
    ax_alpha.set_ylabel(r"Exponent $n$ in $\delta \propto B_0^n$")
    ax_summary.set_title("The prefactor changes more than the exponent")
    ax_summary.grid(True, ls=":", alpha=0.35)
    ax_summary.set_xlim(0.0, 0.5)
    ax_summary.axhline(1.0, color="gray", ls=":", lw=0.8)
    ax_summary.axhline(0.0, color="gray", ls=":", lw=0.8)

    lines1, labels1 = ax_summary.get_legend_handles_labels()
    lines2, labels2 = ax_alpha.get_legend_handles_labels()
    ax_summary.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper left")

    fig.suptitle(
        r"2D wall-to-midplane analysis: the $\sqrt{D h}$ form survives in the bulk,"
        r" while wall-adjacent slices show a suppressed prefactor",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "z_dependence_2d.pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"Saved figure to {FIGURES_DIR / 'z_dependence_2d.pdf'}")
    print(f"Saved metrics to {PROCESSED_DIR / 'z_dependence_2d_metrics.csv'}")


if __name__ == "__main__":
    main()