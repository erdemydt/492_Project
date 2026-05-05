"""scaling_2d.py — Summary scaling analysis for the 2D B-sweep.

This reads processed/falloff_2d_by_z.csv, extracts the midplane slice, and
generates a compact figure showing:
  - δ_{1/2} vs B0 on log-log axes for the midplane
  - the corresponding α = δ / √(Dh) profile across B0
  - a heatmap of δ_{1/2}(B0, z) across the full 2D export

Outputs:
  figures/scaling_2d.pdf
  processed/scaling_2d_metrics.csv
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

R_IN_MM = 0.1
H_MM = 1.0
SIGMA = 3.7e6
MU = 1.95e-3


def hartmann_thickness_m(b0: np.ndarray | float) -> np.ndarray | float:
    return np.sqrt(MU / (SIGMA * np.asarray(b0) ** 2))


def power_law(x: np.ndarray, a: float, n: float) -> np.ndarray:
    return a * x**n


def main() -> None:
    FIGURES_DIR.mkdir(exist_ok=True)

    df = pd.read_csv(PROCESSED_DIR / "falloff_2d_by_z.csv")
    df = df.sort_values(["B0_T", "z_mm"]).reset_index(drop=True)

    midplane_idx = df.groupby("B0_T")["z_mm"].apply(lambda s: (s - 0.5).abs().idxmin())
    midplane = df.loc[midplane_idx].sort_values("B0_T").reset_index(drop=True)
    valid_midplane = midplane.dropna(subset=["delta_half_um"]).copy()

    if valid_midplane.empty:
        raise RuntimeError("No valid midplane fall-off values were found in falloff_2d_by_z.csv")

    b0 = valid_midplane["B0_T"].to_numpy(dtype=float)
    delta_um = valid_midplane["delta_half_um"].to_numpy(dtype=float)
    delta_m = delta_um * 1e-6
    D_m = hartmann_thickness_m(b0)
    sqDh_um = np.sqrt(D_m * (H_MM * 1e-3)) * 1e6
    alpha = delta_m / np.sqrt(D_m * (H_MM * 1e-3))

    popt, pcov = curve_fit(power_law, b0, delta_m, p0=[2e-4, -0.5])
    A_fit, n_fit = popt
    A_err, n_err = np.sqrt(np.diag(pcov))

    alpha_mean = float(np.mean(alpha))
    alpha_median = float(np.median(alpha))
    alpha_rms = float(np.sqrt(np.mean((alpha - alpha_median) ** 2)))

    metrics = valid_midplane[["B0_T", "z_mm", "delta_half_um", "delta_third_um", "bulk_const", "peak_excess"]].copy()
    metrics["D_um"] = D_m * 1e6
    metrics["sqrtDh_um"] = sqDh_um
    metrics["alpha"] = alpha
    metrics.to_csv(PROCESSED_DIR / "scaling_2d_metrics.csv", index=False)

    print("2D midplane scaling summary")
    print("=" * 34)
    print(f"Power-law fit: delta = {A_fit*1e6:.1f} µm * B0^{n_fit:.3f} ± {n_err:.3f}")
    print(f"alpha mean = {alpha_mean:.4f}")
    print(f"alpha median = {alpha_median:.4f}")
    print(f"alpha RMS spread = {alpha_rms:.4f}")

    delta_fit = power_law(b0, A_fit, n_fit)
    B_grid = np.logspace(np.log10(b0.min()), np.log10(b0.max()), 400)
    delta_grid = power_law(B_grid, A_fit, n_fit)
    sqDh_grid = np.sqrt(hartmann_thickness_m(B_grid) * (H_MM * 1e-3)) * 1e6

    heatmap = df.pivot(index="z_mm", columns="B0_T", values="delta_half_um")
    heatmap = heatmap.sort_index(axis=0).sort_index(axis=1)

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(13, 4.8))

    ax_left.loglog(b0, delta_um, "o", ms=5, color="#225ea8", label=r"midplane $\delta_{1/2}$")
    ax_left.loglog(B_grid, delta_grid * 1e6, "k-", lw=1.8, label=rf"fit: $\delta = {A_fit*1e6:.0f}\,\mu m\,B_0^{{{n_fit:.2f}}}$")
    ax_left.loglog(b0, np.sqrt(MU / (SIGMA * b0**2) * (H_MM * 1e-3)) * 1e6,
                   "--", color="#d95f0e", lw=1.6, label=r"$\sqrt{Dh}$")
    ax_left.set_xlabel(r"$B_0$  (T)")
    ax_left.set_ylabel(r"$\delta_{1/2} = r_{1/2} - r_\mathrm{in}$  (µm)")
    ax_left.set_title("2D midplane fall-off scaling")
    ax_left.grid(True, which="both", ls=":", alpha=0.35)
    ax_left.legend(fontsize=8)
    ax_left.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax_left.yaxis.set_major_formatter(ticker.ScalarFormatter())

    if heatmap.notna().any().any():
        mesh = ax_right.pcolormesh(
            heatmap.columns.to_numpy(dtype=float),
            heatmap.index.to_numpy(dtype=float),
            heatmap.to_numpy(dtype=float),
            shading="auto",
            cmap="viridis",
        )
        cb = fig.colorbar(mesh, ax=ax_right)
        cb.set_label(r"$\delta_{1/2}$  (µm)")
    ax_right.set_xlabel(r"$B_0$  (T)")
    ax_right.set_ylabel(r"$z$  (mm)")
    ax_right.set_title(r"$\delta_{1/2}(B_0, z)$ from 2D export")

    fig.suptitle(
        rf"2D B-sweep summary: $\delta_{{1/2}} \propto B_0^{{{n_fit:.2f}}}$  and  $\alpha \approx {alpha_median:.2f}$",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "scaling_2d.pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"Saved figure to {FIGURES_DIR / 'scaling_2d.pdf'}")
    print(f"Saved metrics to {PROCESSED_DIR / 'scaling_2d_metrics.csv'}")


if __name__ == "__main__":
    main()