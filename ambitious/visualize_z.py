"""visualize_z.py — Heatmap of delta(B,z) and distance-from-wall plots.

Generates:
 - ambitious/figures/delta_heatmap_Bz.pdf
 - ambitious/figures/alpha_vs_d.pdf
 - ambitious/figures/exponent_vs_d.pdf

Usage: run from project root: `python ambitious/visualize_z.py`
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
PROCESSED = ROOT / "processed"
FIGS = ROOT / "figures"
FIGS.mkdir(exist_ok=True)

H_MM = 1.0


def main():
    df = pd.read_csv(PROCESSED / "falloff_2d_by_z.csv")
    # bin z to same q grid used in fits
    df["z_q"] = df["z_mm"].round(3)

    # pivot mean delta_half_um for heatmap (rows: z, cols: B)
    pivot = df.pivot_table(index="z_q", columns="B0_T", values="delta_half_um", aggfunc="mean")
    pivot = pivot.sort_index()
    B_vals = pivot.columns.to_numpy(dtype=float)
    z_vals = pivot.index.to_numpy(dtype=float)
    data = pivot.to_numpy()

    # Heatmap
    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.pcolormesh(B_vals, z_vals, data, shading='auto', cmap='viridis')
    c = fig.colorbar(im, ax=ax)
    c.set_label('delta_half (µm)')
    ax.set_xscale('log')
    ax.set_xlabel('B0 (T)')
    ax.set_ylabel('z (mm)')
    ax.set_title('δ_half (µm) heatmap over B and z')
    fig.tight_layout()
    fig.savefig(FIGS / 'delta_heatmap_Bz.pdf')
    plt.close(fig)

    # distance-from-wall plots using z_scaling_by_z
    zres = pd.read_csv(PROCESSED / 'z_scaling_by_z.csv')
    if zres.empty:
        print('No per-z fit results found in processed/z_scaling_by_z.csv')
        return

    zres['d_mm'] = np.minimum(zres['z_mm'], H_MM - zres['z_mm'])
    zres = zres.sort_values('d_mm')

    # alpha vs d
    fig, ax = plt.subplots(figsize=(7,4))
    ax.plot(zres['d_mm'], zres['alpha_med'], 'o-', color='#2b8cbe')
    ax.fill_between(zres['d_mm'], zres['alpha_med']-zres['alpha_std'], zres['alpha_med']+zres['alpha_std'],
                    color='#2b8cbe', alpha=0.2)
    ax.set_xlabel('distance from nearest wall d (mm)')
    ax.set_ylabel(r'median $\,\alpha = \delta/\sqrt{Dh}$')
    ax.set_title('Alpha vs distance-from-wall')
    fig.tight_layout()
    fig.savefig(FIGS / 'alpha_vs_d.pdf')
    plt.close(fig)

    # exponent n vs d
    fig, ax = plt.subplots(figsize=(7,4))
    ax.plot(zres['d_mm'], zres['n_fit'], 'o-', color='#7b3294')
    ax.fill_between(zres['d_mm'], zres['n_fit']-zres['n_err'], zres['n_fit']+zres['n_err'], color='#7b3294', alpha=0.2)
    ax.axhline(-0.5, color='k', ls='--', lw=0.8)
    ax.set_xlabel('distance from nearest wall d (mm)')
    ax.set_ylabel('fitted exponent n')
    ax.set_title('Exponent n vs distance-from-wall')
    fig.tight_layout()
    fig.savefig(FIGS / 'exponent_vs_d.pdf')
    plt.close(fig)

    print('Saved delta heatmap and distance-from-wall plots to ambitious/figures/')


if __name__ == '__main__':
    main()
