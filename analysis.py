"""
analysis.py — MHD Corbino Disk Post-Processing
Produces: jr_profiles.pdf, bulk_validation.pdf, falloff_profiles.pdf
          processed/falloff_data.csv
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os

os.makedirs("figures", exist_ok=True)
os.makedirs("processed", exist_ok=True)

# ── Geometry & material parameters ───────────────────────────────────────────
r_in   = 0.1e-3   # m
r_out  = 4.5e-3   # m
h      = 1.0e-3   # m
sigma  = 3.7e6    # S/m
mu     = 1.95e-3  # Pa·s

r_in_mm  = r_in * 1e3      # 0.1 mm
r_bulk_mm = 3.0 * r_in_mm  # 0.3 mm  (3 × r_in, near-electrode window boundary)

# ── Load & sort data ──────────────────────────────────────────────────────────
df = pd.read_csv(
    "data/BsweepTrialData.txt",
    comment="%", sep=r"\s+",
    names=["r", "z", "B0_param", "Jr", "v", "B0_dup"],
)
df = df.drop(columns=["B0_dup"])
df = df.sort_values(["B0_param", "r"]).reset_index(drop=True)
df["r_m"] = df["r"] * 1e-3   # r in metres

B0_vals = sorted(df["B0_param"].unique())
n_B     = len(B0_vals)

# ── Task 1: Data inspection ───────────────────────────────────────────────────
print("=" * 60)
print("TASK 1 — DATA INSPECTION")
print("=" * 60)
print(f"B0 values : {[float(f'{v:.1f}') for v in B0_vals]}")
print(f"Total rows: {len(df)}")
print(f"r range   : {df['r'].min():.4f} – {df['r'].max():.4f} mm")
print(f"\n{'B0 (T)':>8}  {'rows':>6}  {'r_min (mm)':>12}  {'r_max (mm)':>12}")
print("-" * 44)
for B0 in B0_vals:
    sub = df[df["B0_param"] == B0]
    print(f"{B0:>8.1f}  {len(sub):>6}  {sub['r'].min():>12.4f}  {sub['r'].max():>12.4f}")

# ── Colourmap helper ──────────────────────────────────────────────────────────
cmap   = cm.viridis
colors = [cmap(i / (n_B - 1)) for i in range(n_B)]

def colorbar(fig, ax, label):
    sm = plt.cm.ScalarMappable(
        cmap=cmap, norm=plt.Normalize(vmin=min(B0_vals), vmax=max(B0_vals)))
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax)
    cb.set_label(label, fontsize=10)
    return cb

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1 — J_r(r) vs r  (narrative starting point)
# ─────────────────────────────────────────────────────────────────────────────
# Show 5 representative B₀ values with distinct labels so the audience can
# immediately read off the effect of B on the current profile.
SHOW_B0 = [0.5, 2.0, 5.0, 10.0, 14.0]
show_colors = plt.cm.plasma(np.linspace(0.15, 0.85, len(SHOW_B0)))

fig1, ax1 = plt.subplots(figsize=(7, 4))

for B0, col in zip(SHOW_B0, show_colors):
    sub = df[df["B0_param"] == B0].copy()
    ax1.loglog(sub["r"], sub["Jr"], color=col, lw=1.6,
               label=f"$B_0 = {B0:.1f}$ T")

# 1/r reference line anchored to B0=0.5T bulk value at r=1 mm
sub_ref = df[df["B0_param"] == 0.5]
J_ref   = float(sub_ref[sub_ref["r"].between(0.95, 1.05)]["Jr"].median())
r_ref   = np.array([0.1, 3.0])
ax1.loglog(r_ref, J_ref * 1.0 / r_ref, "k--", lw=1.2, label=r"$\propto 1/r$")

ax1.axvline(r_in_mm, color="gray", lw=0.8, ls=":", alpha=0.8)
ax1.text(r_in_mm * 1.12, ax1.get_ylim()[0] * 3, r"$r_\mathrm{in}$",
         fontsize=8, color="gray", va="bottom")

ax1.set_xlabel("$r$  (mm)", fontsize=11)
ax1.set_ylabel("$J_r$  (A m$^{-2}$)", fontsize=11)
ax1.set_title("Radial current density profiles at midplane ($z = h/2$)\n"
              "Near $r_\\mathrm{in}$ the current exceeds the bulk $1/r$ form",
              fontsize=10)
ax1.legend(fontsize=8, loc="lower left")
ax1.grid(True, which="both", ls=":", alpha=0.35)
fig1.tight_layout()
fig1.savefig("figures/jr_profiles.pdf")
plt.close(fig1)
print("\nSaved: figures/jr_profiles.pdf")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2 — J_r·r vs r  (bulk validation + excess hump)
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("TASK 2 — BULK VALIDATION")
print("=" * 60)

bulk_consts = {}

fig2, ax2 = plt.subplots(figsize=(7, 4))

for idx, B0 in enumerate(B0_vals):
    sub = df[df["B0_param"] == B0].copy()
    sub["Jr_r"] = sub["Jr"] * sub["r_m"]
    bulk_mask   = sub["r"] > r_bulk_mm
    bulk_const  = np.median(sub.loc[bulk_mask, "Jr_r"])
    bulk_consts[B0] = bulk_const
    ax2.plot(sub["r"], sub["Jr_r"], color=colors[idx], lw=1.1)

ax2.axvline(r_bulk_mm, color="k", ls="--", lw=0.9,
            label=f"$r = 3r_\\mathrm{{in}} = {r_bulk_mm:.2f}$ mm")
ax2.set_xlabel("$r$  (mm)", fontsize=11)
ax2.set_ylabel(r"$J_r \cdot r$  (A m$^{-1}$)", fontsize=11)
ax2.set_title(r"$J_r \cdot r$ versus $r$  —  flat in bulk, elevated near $r_\mathrm{in}$"
              "\n(each curve is one $B_0$; colour = $B_0$)", fontsize=10)
colorbar(fig2, ax2, "$B_0$  (T)")
ax2.legend(fontsize=8)
fig2.tight_layout()
fig2.savefig("figures/bulk_validation.pdf")
plt.close(fig2)
print("Saved: figures/bulk_validation.pdf")

for B0 in B0_vals:
    print(f"  B0={B0:5.1f} T  bulk_const = {bulk_consts[B0]:.4e} A/m")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 3 — Normalised excess fall-off profiles
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("TASK 3 — FALL-OFF PROFILES")
print("=" * 60)

records = []

fig3, ax3 = plt.subplots(figsize=(7, 4))

for idx, B0 in enumerate(B0_vals):
    sub = df[df["B0_param"] == B0].copy()
    sub["Jr_r"]  = sub["Jr"] * sub["r_m"]
    bulk_const   = bulk_consts[B0]
    sub["excess"] = sub["Jr_r"] - bulk_const

    # Peak in near-electrode window
    near       = sub[sub["r"] < r_bulk_mm]
    if near.empty or near["excess"].max() <= 0:
        print(f"  WARNING: B0={B0:.1f} T — no positive excess, skipping")
        continue
    peak_excess = near["excess"].max()
    r_peak_mm   = sub.loc[near["excess"].idxmax(), "r"]

    # Decay region starts at peak, extends full r range
    decay = sub[sub["r"] >= r_peak_mm].copy()

    # Region of interest: excess > 5% of peak
    roi = decay[decay["excess"] > 0.05 * peak_excess]
    if len(roi) < 2:
        print(f"  WARNING: B0={B0:.1f} T — too few ROI points, skipping")
        continue

    r_grid_mm   = np.linspace(roi["r"].iloc[0], roi["r"].iloc[-1], 500)
    excess_grid = np.interp(r_grid_mm, decay["r"].values, decay["excess"].values)

    def find_crossing(frac):
        mask = excess_grid <= frac * peak_excess
        return r_grid_mm[np.argmax(mask)] if mask.any() else roi["r"].iloc[-1]

    r_half_mm  = find_crossing(0.50)
    r_third_mm = find_crossing(0.33)

    records.append({
        "B0":          B0,
        "bulk_const":  bulk_const,
        "peak_excess": peak_excess,
        "r_half_mm":   r_half_mm,
        "r_third_mm":  r_third_mm,
    })
    print(f"  B0={B0:5.1f} T  peak={peak_excess:.4f}  "
          f"r_half={(r_half_mm - r_in_mm)*1e3:6.1f} µm  "
          f"r_third={(r_third_mm - r_in_mm)*1e3:6.1f} µm")

    # Plot: normalised excess vs distance from r_in (µm)
    x_um   = (r_grid_mm - r_in_mm) * 1e3
    y_norm = excess_grid / peak_excess
    ax3.plot(x_um, y_norm, color=colors[idx], lw=1.1)

    # Mark r_half on each curve
    x_half_um = (r_half_mm - r_in_mm) * 1e3
    ax3.plot(x_half_um, 0.5, "|", color=colors[idx], ms=8, mew=1.5)

ax3.axhline(0.50, color="k",    lw=0.9, ls="--", label=r"50 %  ($r_{1/2}$)")
ax3.axhline(0.33, color="gray", lw=0.9, ls=":",  label=r"33 %  ($r_{1/3}$)")
ax3.set_xlabel(r"$r - r_\mathrm{in}$  (µm)", fontsize=11)
ax3.set_ylabel(r"$(J_r r - C_\mathrm{bulk})\;/\;\epsilon_\mathrm{peak}$", fontsize=11)
ax3.set_title("Normalised excess current near the inner electrode\n"
              r"Tick marks show $r_{1/2}$ for each $B_0$  (colour = $B_0$)",
              fontsize=10)
colorbar(fig3, ax3, "$B_0$  (T)")
ax3.legend(fontsize=9, loc="upper right")
fig3.tight_layout()
fig3.savefig("figures/falloff_profiles.pdf")
plt.close(fig3)
print("Saved: figures/falloff_profiles.pdf")

# ── Save processed CSV ────────────────────────────────────────────────────────
results_df = pd.DataFrame(records)
results_df.to_csv("processed/falloff_data.csv", index=False)
print("\nSaved: processed/falloff_data.csv")
print(results_df.to_string(index=False))
