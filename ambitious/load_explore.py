"""
load_explore.py — Step 1: Load BSweep2D.txt, report mesh structure, basic sanity checks.
Run this first; nothing is saved except console output.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_FILE = "../data_2D/BSweep2D.txt"
ROUND_DECIMALS = 5

# ── Load ──────────────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv(DATA_FILE, comment="%", sep=r"\s+",
                 names=["r", "z", "B0", "Jr"])
df["r_q"] = df["r"].round(ROUND_DECIMALS)
df["z_q"] = df["z"].round(ROUND_DECIMALS)
df = df.sort_values(["B0", "z_q", "r_q"]).reset_index(drop=True)
print(f"  Total rows: {len(df):,}")

# ── B0 values ─────────────────────────────────────────────────────────────────
B0_vals = sorted(df["B0"].unique())
print(f"\nB0 values ({len(B0_vals)}): {B0_vals}")

# ── Per-B0 mesh report ────────────────────────────────────────────────────────
print("\nPer-B0 mesh structure:")
print(f"{'B0':>6}  {'rows':>8}  {'groups':>8}  {'uniq_r':>7}  {'uniq_z':>7}  {'dup':>6}  {'r_min(mm)':>10}  {'r_max(mm)':>10}  {'z_min(mm)':>10}  {'z_max(mm)':>10}")
for b in B0_vals:
    sl = df[df["B0"] == b]
    groups = sl.groupby(["r_q", "z_q"]).size()
    ur = sl["r_q"].nunique()
    uz = sl["z_q"].nunique()
    dup = int((groups > 1).sum())
    print(f"{b:>6.1f}  {len(sl):>8,}  {len(groups):>8,}  {ur:>7}  {uz:>7}  {dup:>6d}  "
          f"{sl['r'].min():>10.4f}  {sl['r'].max():>10.4f}  "
          f"{sl['z'].min():>10.4f}  {sl['z'].max():>10.4f}")

# ── Check if mesh is structured after rounding ────────────────────────────────
print("\nChecking rounded mesh structure for B0=0.5 T...")
sl = df[df["B0"] == 0.5].copy()
ur = sl["r_q"].nunique()
uz = sl["z_q"].nunique()
groups = sl.groupby(["r_q", "z_q"]).size()
n_expected = ur * uz
n_actual = len(groups)
print(f"  unique rounded r: {ur},  unique rounded z: {uz}")
print(f"  expected groups if tensor-product: {n_expected:,}")
print(f"  actual rounded groups: {n_actual:,}")
if n_expected == n_actual:
    print("  => NEARLY STRUCTURED after rounding — pivot will work with duplicate aggregation")
else:
    print(f"  => PARTIALLY STRUCTURED — {n_actual / n_expected:.1%} of full tensor product")
    z_per_r = sl.groupby("r_q")["z_q"].count()
    print(f"  z values per rounded r: min={z_per_r.min()}, max={z_per_r.max()}, "
          f"mean={z_per_r.mean():.1f}, median={z_per_r.median():.0f}")
    r_per_z = sl.groupby("z_q")["r_q"].count()
    print(f"  r values per rounded z: min={r_per_z.min()}, max={r_per_z.max()}, "
          f"mean={r_per_z.mean():.1f}, median={r_per_z.median():.0f}")

# ── z spacing statistics ──────────────────────────────────────────────────────
z_vals = np.sort(sl["z_q"].unique())
dz = np.diff(z_vals)
print(f"\nz spacing (B0=0.5): min={dz.min()*1e3:.2f} µm, max={dz.max()*1e3:.2f} µm, "
      f"mean={dz.mean()*1e3:.2f} µm")

# ── Hartmann layer thickness vs z spacing ────────────────────────────────────
mu    = 1.95e-3
sigma = 3.7e6
print("\nHartmann layer thickness D vs z-mesh spacing:")
dz_mean = dz.mean()
for b in [0.5, 1, 2, 5, 10, 14]:
    D = np.sqrt(mu / (sigma * b**2)) * 1e3  # mm
    print(f"  B0={b:4.1f} T:  D = {D*1e3:.2f} µm,  dz_mean ≈ {dz_mean*1e3:.2f} µm,  "
          f"ratio D/dz = {D/dz_mean:.1f}")

# ── Jr range sanity check ────────────────────────────────────────────────────
print("\nJr statistics per B0:")
print(f"{'B0':>6}  {'Jr_min':>12}  {'Jr_max':>12}  {'Jr_at_r_in_z0':>16}")
for b in B0_vals[:6]:
    sl = df[df["B0"] == b]
    # Jr at r≈r_in, z≈0
    near_in = sl[sl["r"] < 0.11].iloc[0]
    print(f"{b:>6.1f}  {sl['Jr'].min():>12.2f}  {sl['Jr'].max():>12.2f}  {near_in['Jr']:>16.4f}")

print("\nDone. No files written.")
