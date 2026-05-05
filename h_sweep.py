"""
h_sweep.py — MHD Corbino Disk: H-sweep post-processing (B0 = 5 T fixed)

The H-sweep data has a more complex Jr(r) structure than the B-sweep:
  - Near r_in: elevated Jr (current redistribution from electrode)
  - Intermediate r: NEGATIVE Jr at midplane (Hartmann return current;
    the net radial current is carried in the z-boundary layers, leaving
    the bulk midplane with reverse current)
  - Near r_out: Jr rises positive again (current exits into outer electrode)

Analysis strategy:
  - C_bulk = 0  (midplane bulk current ≈ 0 at B0=5T, all Ha values here are large)
  - excess(r) = Jr(r) * r   (i.e. the profile itself)
  - Decay search stops at first zero crossing of Jr*r to avoid outer-electrode region
  - r_half = first r where excess drops to 50% of peak

Produces:
  figures/jr_profiles_h.pdf       — raw profiles (selected h), showing return current
  figures/falloff_profiles_h.pdf  — normalised near-electrode decay for all h
  processed/falloff_data_h.csv    — delta_half, delta_third per h value
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.ticker as ticker
import os

os.makedirs("figures", exist_ok=True)
os.makedirs("processed", exist_ok=True)

# ── Constants ─────────────────────────────────────────────────────────────────
r_in   = 0.1e-3    # m
sigma  = 3.7e6     # S/m
mu     = 1.95e-3   # Pa·s
B0     = 5.0       # T  (confirmed from Jr*r at r_in matching B-sweep at B0=5T)
D      = np.sqrt(mu / (sigma * B0**2))   # Hartmann layer thickness at B0=5T

r_in_mm   = r_in * 1e3       # 0.1 mm
r_near_mm = 3.0 * r_in_mm    # 0.3 mm  (near-electrode window for peak search)

print("=" * 60)
print("H-SWEEP ANALYSIS  (B₀ = 5 T confirmed)")
print("=" * 60)
print(f"D(5T) = {D*1e6:.3f} µm    Ha(h=1mm) = {B0*1e-3*np.sqrt(sigma/mu):.1f}")

# ── Load & sort ───────────────────────────────────────────────────────────────
df = pd.read_csv(
    "data/HsweepTrialData.txt",
    comment="%", sep=r"\s+",
    names=["r", "z", "h_param", "Jr", "v", "h_dup"],
)
df = df.drop(columns=["h_dup"])
df = df.sort_values(["h_param", "r"]).reset_index(drop=True)
df["r_m"] = df["r"] * 1e-3

h_vals = sorted(df["h_param"].unique())
n_h    = len(h_vals)

print(f"\n{n_h} h values: {[round(v*1e3,1) for v in h_vals]} mm")
print(f"Rows per h: {len(df[df['h_param']==h_vals[0]])}  (constant — fixed mesh)")

# ── Colourmap ─────────────────────────────────────────────────────────────────
cmap   = cm.plasma
colors = [cmap(i / (n_h - 1)) for i in range(n_h)]

def add_colorbar(fig, ax, vmin, vmax, label):
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax)
    cb.set_label(label, fontsize=10)

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1 — J_r(r) vs r for representative h values
#            Shows: (left) near-electrode region,  (right) full profile
# ─────────────────────────────────────────────────────────────────────────────
SHOW_H_mm = [0.5, 1.0, 3.0, 7.0, 15.0]
show_cols  = plt.cm.viridis(np.linspace(0.1, 0.9, len(SHOW_H_mm)))

fig1, (ax_near, ax_full) = plt.subplots(1, 2, figsize=(13, 4.5))

for h_mm, col in zip(SHOW_H_mm, show_cols):
    h_val = h_mm * 1e-3
    sub   = df[np.isclose(df["h_param"], h_val, rtol=1e-4)].copy()
    Ha    = B0 * h_val * np.sqrt(sigma / mu)
    lbl   = f"$h = {h_mm:.0f}$ mm  (Ha = {Ha:.0f})"

    # Left: near-electrode (r < 1 mm)
    near = sub[sub["r"] < 1.0]
    ax_near.plot(near["r"], near["Jr"], color=col, lw=1.6, label=lbl)

    # Right: full profile
    ax_full.plot(sub["r"], sub["Jr"], color=col, lw=1.4, label=lbl)

ax_near.axvline(r_in_mm, color="gray", lw=0.8, ls=":", alpha=0.7)
ax_near.axhline(0,       color="gray", lw=0.8, ls="--", alpha=0.5)
ax_near.set_xlabel("$r$  (mm)", fontsize=11)
ax_near.set_ylabel("$J_r$  (A m$^{-2}$)", fontsize=11)
ax_near.set_title("Near inner electrode  ($r < 1$ mm)", fontsize=10)
ax_near.legend(fontsize=7.5, loc="upper right")
ax_near.grid(ls=":", alpha=0.35)

ax_full.axhline(0, color="gray", lw=0.8, ls="--", alpha=0.5, label="$J_r = 0$")
ax_full.set_xlabel("$r$  (mm)", fontsize=11)
ax_full.set_ylabel("$J_r$  (A m$^{-2}$)", fontsize=11)
ax_full.set_title("Full radial extent  ($B_0 = 5$ T fixed)", fontsize=10)
ax_full.legend(fontsize=7.5, loc="upper right")
ax_full.grid(ls=":", alpha=0.35)

fig1.suptitle(
    r"Midplane $J_r(r)$ for H-sweep  —  "
    r"negative region = Hartmann return current (net $J_r$ carried in wall layers)",
    fontsize=10, y=1.01
)
fig1.tight_layout()
fig1.savefig("figures/jr_profiles_h.pdf", bbox_inches="tight")
plt.close(fig1)
print("\nSaved: figures/jr_profiles_h.pdf")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2 — Normalised excess fall-off + CSV
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("FALL-OFF ANALYSIS")
print("=" * 60)

records = []

fig2, ax2 = plt.subplots(figsize=(7, 4))

for idx, h_val in enumerate(h_vals):
    h_mm = h_val * 1e3
    Ha   = B0 * h_val * np.sqrt(sigma / mu)
    sub  = df[np.isclose(df["h_param"], h_val, rtol=1e-4)].copy()
    sub["Jr_r"] = sub["Jr"] * sub["r_m"]

    # C_bulk = 0 (Hartmann-screened midplane at B0=5T for all Ha here)
    # excess = Jr * r  (the profile itself)
    sub["excess"] = sub["Jr_r"]   # C_bulk ≡ 0

    # Peak in near-electrode window
    near = sub[sub["r"] < r_near_mm]
    if near.empty or near["excess"].max() <= 0:
        print(f"  SKIP h={h_mm:.1f}mm — no positive excess near electrode")
        continue
    peak_excess = near["excess"].max()
    r_peak_mm   = sub.loc[near["excess"].idxmax(), "r"]

    # Decay region: from peak outward, up to first zero crossing of excess
    decay = sub[sub["r"] >= r_peak_mm].copy()
    zero_cross = decay[decay["excess"] <= 0]
    r_stop_mm  = zero_cross["r"].iloc[0] if not zero_cross.empty else decay["r"].iloc[-1]
    decay = decay[decay["r"] <= r_stop_mm]

    if len(decay) < 4:
        print(f"  SKIP h={h_mm:.1f}mm — too few points before zero crossing")
        continue

    # Interpolate onto 500-point grid
    r_grid_mm   = np.linspace(decay["r"].iloc[0], decay["r"].iloc[-1], 500)
    excess_grid = np.interp(r_grid_mm, decay["r"].values, decay["excess"].values)
    excess_grid = np.maximum(excess_grid, 0)   # clamp to ≥ 0 within decay region

    def find_crossing(frac):
        mask = excess_grid <= frac * peak_excess
        return r_grid_mm[np.argmax(mask)] if mask.any() else decay["r"].iloc[-1]

    r_half_mm  = find_crossing(0.50)
    r_third_mm = find_crossing(0.33)

    delta_half_um  = (r_half_mm  - r_in_mm) * 1e3
    delta_third_um = (r_third_mm - r_in_mm) * 1e3

    records.append({
        "h_mm":        h_mm,
        "h_m":         h_val,
        "Ha":          Ha,
        "peak_excess": peak_excess,
        "r_half_mm":   r_half_mm,
        "r_third_mm":  r_third_mm,
        "delta_half_um":  delta_half_um,
        "delta_third_um": delta_third_um,
        "sqDh_um":     np.sqrt(D * h_val) * 1e6,
        "r_stop_mm":   r_stop_mm,
    })

    print(f"  h={h_mm:5.1f}mm  Ha={Ha:6.1f}  "
          f"δ_half={delta_half_um:6.1f}µm  "
          f"δ_third={delta_third_um:6.1f}µm  "
          f"√(Dh)={np.sqrt(D*h_val)*1e6:5.1f}µm  "
          f"α={delta_half_um/1e6/np.sqrt(D*h_val):.3f}")

    # Normalised excess vs (r - r_in) in µm
    x_um   = (r_grid_mm - r_in_mm) * 1e3
    y_norm = excess_grid / peak_excess
    ax2.plot(x_um, y_norm, color=colors[idx], lw=1.0)
    # Tick at r_half
    ax2.plot((r_half_mm - r_in_mm) * 1e3, 0.5, "|",
             color=colors[idx], ms=7, mew=1.4)

ax2.axhline(0.50, color="k",    lw=0.9, ls="--", label=r"50 %  ($r_{1/2}$)")
ax2.axhline(0.33, color="gray", lw=0.9, ls=":",  label=r"33 %  ($r_{1/3}$)")
ax2.set_xlabel(r"$r - r_\mathrm{in}$  (µm)", fontsize=11)
ax2.set_ylabel(r"$J_r \cdot r \;/\; (J_r r)_\mathrm{peak}$", fontsize=11)
ax2.set_title("H-sweep: normalised excess near inner electrode  ($B_0 = 5$ T)\n"
              r"Tick marks = $r_{1/2}$; colour = $h$ from 0.5 mm (dark) to 15 mm (light)",
              fontsize=10)
add_colorbar(fig2, ax2, h_vals[0]*1e3, h_vals[-1]*1e3, "$h$  (mm)")
ax2.legend(fontsize=9, loc="upper right")
fig2.tight_layout()
fig2.savefig("figures/falloff_profiles_h.pdf")
plt.close(fig2)
print("\nSaved: figures/falloff_profiles_h.pdf")

# ── Save processed CSV ────────────────────────────────────────────────────────
results_df = pd.DataFrame(records)
results_df.to_csv("processed/falloff_data_h.csv", index=False)
print("Saved: processed/falloff_data_h.csv")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 3 — δ vs h: linear + log-log with fit
# ─────────────────────────────────────────────────────────────────────────────
from scipy.optimize import curve_fit

h_arr     = results_df["h_m"].values
delta_arr = results_df["delta_half_um"].values * 1e-6   # m
sqDh_arr  = results_df["sqDh_um"].values * 1e-6         # m

def power_law(x, A, n):
    return A * x**n

popt, pcov = curve_fit(power_law, h_arr, delta_arr, p0=[1e-4, 0.5])
A_h, n_h   = popt
n_h_err    = np.sqrt(np.diag(pcov))[1]
alpha_h    = float(np.median(delta_arr / sqDh_arr))

print()
print("=" * 60)
print("H-SWEEP SCALING FIT")
print("=" * 60)
print(f"δ_half = A · h^n")
print(f"  A = {A_h*1e6:.1f} µm·m^(-n)")
print(f"  n = {n_h:.3f} ± {n_h_err:.3f}")
print(f"  α = δ / √(D·h) = {alpha_h:.3f}  (cf. B-sweep α = 1.406)")

h_fine  = np.linspace(h_arr.min(), h_arr.max(), 400)
h_fine_l = np.logspace(np.log10(h_arr.min()), np.log10(h_arr.max()), 400)
fit_fine  = power_law(h_fine,   A_h, n_h)
fit_fine_l = power_law(h_fine_l, A_h, n_h)
sqDh_fine = np.sqrt(D * h_fine_l)

fit_lbl = (rf"Fit: $\delta_{{1/2}} = {A_h*1e6:.0f}\,\mu\mathrm{{m}}"
           rf"\cdot h^{{{n_h:.2f}}}$")
th_lbl  = rf"$\alpha\sqrt{{Dh}}$ $(\alpha={alpha_h:.2f},\;h^{{0.5}})$"

fig3, (ax_lin, ax_log) = plt.subplots(1, 2, figsize=(13, 4.5))

for ax, h_x, fit_x, is_log in [
        (ax_lin, h_arr*1e3,   fit_fine*1e6,   False),
        (ax_log, h_arr*1e3,   fit_fine_l*1e6, True)]:
    ax.scatter(h_arr*1e3, delta_arr*1e6,
               color="royalblue", s=30, zorder=5,
               label=r"$\delta_{1/2}$  (data)")
    ax.scatter(h_arr*1e3, results_df["delta_third_um"],
               color="cornflowerblue", s=18, marker="s", alpha=0.65,
               zorder=4, label=r"$\delta_{1/3}$  (data)")
    if is_log:
        ax.loglog(h_fine_l*1e3, fit_fine_l*1e6, "k-",  lw=2.0, label=fit_lbl)
        ax.loglog(h_fine_l*1e3, alpha_h*sqDh_fine*1e6, "r--", lw=1.8, label=th_lbl)
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.grid(True, which="both", ls=":", alpha=0.35)
        ax.set_title("Log-log: confirms $h^{1/2}$ power law", fontsize=10)
    else:
        ax.plot(h_fine*1e3,   fit_fine*1e6,          "k-",  lw=2.0, label=fit_lbl)
        ax.plot(h_fine_l*1e3, alpha_h*sqDh_fine*1e6, "r--", lw=1.8, label=th_lbl)
        ax.set_xlim(left=0); ax.set_ylim(bottom=0)
        ax.grid(ls=":", alpha=0.4)
        ax.set_title("Linear axes", fontsize=10)
    ax.set_xlabel("$h$  (mm)", fontsize=11)
    ax.set_ylabel(r"$\delta_{1/2} = r_{1/2} - r_\mathrm{in}$  (µm)", fontsize=11)
    ax.legend(fontsize=8.5, loc="upper left")

fig3.suptitle(
    rf"$\delta_{{1/2}}$ vs fluid height $h$  at fixed $B_0 = 5$ T"
    rf"  —  fitted exponent $n = {n_h:.2f} \pm {n_h_err:.2f}$  (expected $+1/2$)",
    fontsize=11, y=1.01
)
fig3.tight_layout()
fig3.savefig("figures/scaling_h.pdf", bbox_inches="tight")
plt.close(fig3)
print("Saved: figures/scaling_h.pdf")
