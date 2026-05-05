"""
combined_scaling.py — Definitive scaling collapse

Both B-sweep (varying B₀, fixed h = 1 mm) and H-sweep (varying h, fixed B₀ = 5 T)
plotted as δ_half vs √(D·h) on the same axes. If δ = α √(Dh), both datasets must
collapse onto a single straight line regardless of which parameter was varied.

This is the proof that the governing scale is √(Dh), not D or h separately.

Produces:
  figures/combined_collapse.pdf  — the definitive figure
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.ticker as ticker
from scipy.optimize import curve_fit
import os

os.makedirs("figures", exist_ok=True)

# ── Constants ─────────────────────────────────────────────────────────────────
r_in  = 0.1e-3
h_B   = 1.0e-3      # fixed h in B-sweep
B0_H  = 5.0         # fixed B0 in H-sweep  (confirmed)
sigma = 3.7e6
mu    = 1.95e-3

def D_func(B):
    return np.sqrt(mu / (sigma * B**2))

# ── Load processed results ────────────────────────────────────────────────────
dfB = pd.read_csv("processed/falloff_data.csv")      # B-sweep
dfH = pd.read_csv("processed/falloff_data_h.csv")    # H-sweep

# B-sweep: D varies with B0, h = 1 mm fixed
B0_arr     = dfB["B0"].values
D_B        = D_func(B0_arr)
sqDh_B     = np.sqrt(D_B * h_B)                             # m
delta_B    = (dfB["r_half_mm"].values - r_in*1e3) * 1e-3    # m
delta_B3   = (dfB["r_third_mm"].values - r_in*1e3) * 1e-3

# H-sweep: D = D(5T) fixed, h varies
h_arr      = dfH["h_m"].values
D_H        = D_func(B0_H)                                   # scalar
sqDh_H     = np.sqrt(D_H * h_arr)                           # m
delta_H    = dfH["delta_half_um"].values * 1e-6             # m
delta_H3   = dfH["delta_third_um"].values * 1e-6

# ── Cross-validation point (same B0=5T, h=1mm) ────────────────────────────────
sqDh_cross = np.sqrt(D_func(B0_H) * h_B)
# B-sweep at B0=5T
idx_5T = np.argmin(np.abs(B0_arr - B0_H))
delta_cross_B = delta_B[idx_5T]
# H-sweep at h=1mm
idx_1mm = np.argmin(np.abs(h_arr - h_B))
delta_cross_H = delta_H[idx_1mm]
print(f"Cross-validation (B0=5T, h=1mm):")
print(f"  B-sweep: √(Dh)={sqDh_cross*1e6:.1f}µm  δ={delta_cross_B*1e6:.1f}µm")
print(f"  H-sweep: √(Dh)={sqDh_cross*1e6:.1f}µm  δ={delta_cross_H*1e6:.1f}µm")
print(f"  Agreement: {abs(delta_cross_B-delta_cross_H)/delta_cross_B*100:.1f}%")

# ── Combined fit: δ = α √(Dh) (forced slope 1, fit only α) ───────────────────
all_sqDh  = np.concatenate([sqDh_B,  sqDh_H])
all_delta = np.concatenate([delta_B, delta_H])

alpha_combined = float(np.sum(all_delta * all_sqDh) / np.sum(all_sqDh**2))
residuals = all_delta - alpha_combined * all_sqDh
rms_frac  = np.std(residuals) / np.mean(all_delta)

# Also fit a free power law to the combined set
def power_law(x, A, n):
    return A * x**n

popt, pcov = curve_fit(power_law, all_sqDh, all_delta, p0=[1.4, 1.0])
A_comb, n_comb = popt
n_err = np.sqrt(np.diag(pcov))[1]

print(f"\nCombined fit (both sweeps, free power law):")
print(f"  δ = {A_comb:.4f} · √(Dh)^{n_comb:.4f}")
print(f"  n = {n_comb:.4f} ± {n_err:.4f}  (expected: 1.0)")
print(f"\nForced-slope α (δ = α √(Dh)):")
print(f"  α = {alpha_combined:.4f}")
print(f"  RMS scatter / mean(δ) = {rms_frac*100:.2f}%")

# Per-sweep α
alpha_B = float(np.median(delta_B / sqDh_B))
alpha_H = float(np.median(delta_H / sqDh_H))
print(f"\n  α (B-sweep) = {alpha_B:.4f}")
print(f"  α (H-sweep) = {alpha_H:.4f}")

# ── Range for reference line ──────────────────────────────────────────────────
sqDh_min = min(sqDh_B.min(), sqDh_H.min()) * 0.9
sqDh_max = max(sqDh_B.max(), sqDh_H.max()) * 1.05
sqDh_line = np.linspace(sqDh_min, sqDh_max, 400)

# ── Colouring: B-sweep by B0, H-sweep by h ───────────────────────────────────
cmap_B = cm.Blues_r
cmap_H = cm.Oranges
norm_B = plt.Normalize(vmin=B0_arr.min(), vmax=B0_arr.max())
norm_H = plt.Normalize(vmin=h_arr.min()*1e3, vmax=h_arr.max()*1e3)
col_B  = [cmap_B(norm_B(B)) for B in B0_arr]
col_H  = [cmap_H(norm_H(h*1e3)) for h in h_arr]

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE — Combined collapse: two panels
#   Left : linear axes, all data + universal line
#   Right: log-log axes, all data + universal line
# ─────────────────────────────────────────────────────────────────────────────
fig, (ax_lin, ax_log) = plt.subplots(1, 2, figsize=(14, 5))

for ax, is_log in [(ax_lin, False), (ax_log, True)]:
    # Universal line
    y_line = alpha_combined * sqDh_line
    if is_log:
        ax.loglog(sqDh_line*1e6, y_line*1e6, "k-", lw=2.2, zorder=2,
                  label=rf"$\delta = {alpha_combined:.2f}\,\sqrt{{Dh}}$  (combined fit)")
    else:
        ax.plot(sqDh_line*1e6, y_line*1e6, "k-", lw=2.2, zorder=2,
                label=rf"$\delta = {alpha_combined:.2f}\,\sqrt{{Dh}}$  (combined fit)")

    # B-sweep points (circles), coloured by B0
    for i in range(len(B0_arr)):
        kw = dict(color=col_B[i], zorder=5, s=28, marker="o", linewidths=0.3,
                  edgecolors="k")
        if is_log:
            ax.scatter(sqDh_B[i]*1e6, delta_B[i]*1e6, **kw,
                       label="B-sweep" if i == 0 else "")
        else:
            ax.scatter(sqDh_B[i]*1e6, delta_B[i]*1e6, **kw,
                       label="B-sweep  (vary $B_0$, $h=1$ mm)" if i == 0 else "")

    # H-sweep points (squares), coloured by h
    for i in range(len(h_arr)):
        kw = dict(color=col_H[i], zorder=5, s=28, marker="s", linewidths=0.3,
                  edgecolors="k")
        if is_log:
            ax.scatter(sqDh_H[i]*1e6, delta_H[i]*1e6, **kw,
                       label="H-sweep" if i == 0 else "")
        else:
            ax.scatter(sqDh_H[i]*1e6, delta_H[i]*1e6, **kw,
                       label=r"H-sweep  (vary $h$, $B_0=5$ T)" if i == 0 else "")

    # Mark cross-validation point
    ax.scatter(sqDh_cross*1e6, delta_cross_B*1e6, marker="*", s=200,
               color="gold", edgecolors="k", linewidths=0.8, zorder=9,
               label=r"$B_0=5$ T, $h=1$ mm  (both sweeps)")

    ax.set_xlabel(r"$\sqrt{D(B_0)\,h}$  (µm)", fontsize=12)
    ax.set_ylabel(r"$\delta_{1/2} = r_{1/2} - r_\mathrm{in}$  (µm)", fontsize=12)

    if is_log:
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.grid(True, which="both", ls=":", alpha=0.35)
        ax.set_title("Log-log  (both sweeps on one line → universal scaling)", fontsize=10)
        ax.legend(fontsize=8, loc="upper left")
    else:
        ax.set_xlim(left=0); ax.set_ylim(bottom=0)
        ax.grid(ls=":", alpha=0.4)
        ax.set_title("Linear axes  (both sweeps on one line → universal scaling)", fontsize=10)
        ax.legend(fontsize=8, loc="upper left")

# Colorbars on the right of the full figure
fig.subplots_adjust(right=0.82)

ax_cb_B = fig.add_axes([0.84, 0.54, 0.012, 0.36])
ax_cb_H = fig.add_axes([0.84, 0.12, 0.012, 0.36])

sm_B = plt.cm.ScalarMappable(cmap=cmap_B, norm=norm_B)
sm_B.set_array([])
cb_B = fig.colorbar(sm_B, cax=ax_cb_B)
cb_B.set_label("$B_0$  (T)", fontsize=9)

sm_H = plt.cm.ScalarMappable(cmap=cmap_H, norm=norm_H)
sm_H.set_array([])
cb_H = fig.colorbar(sm_H, cax=ax_cb_H)
cb_H.set_label("$h$  (mm)", fontsize=9)

fig.suptitle(
    rf"Universal scaling collapse:  $\delta_{{1/2}} = \alpha\,\sqrt{{D\,h}}$   "
    rf"with $\alpha = {alpha_combined:.3f}$"
    "\n"
    rf"B-sweep (circles, vary $B_0$) and H-sweep (squares, vary $h$) "
    rf"on one line confirms both exponents simultaneously",
    fontsize=11, y=1.03,
)

fig.savefig("figures/combined_collapse.pdf", bbox_inches="tight")
plt.close(fig)
print("\nSaved: figures/combined_collapse.pdf")
