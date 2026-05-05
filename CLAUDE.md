# CLAUDE.md — MHD Corbino Disk Post-Processing

Read `Aims.md` for full physical and scientific context before writing any code.

---

## Project

Post-processing and analysis of COMSOL FEM simulations of liquid gallium in a Corbino disk
driven by radial current in an axial magnetic field. The goal is to characterise how the
radial current density J_r(r) deviates from its bulk 1/r form near the inner electrode, and
how the fall-off length scales with B and h.

---

## Environment

- Python 3, numpy, pandas, matplotlib, scipy
- All scripts in the project root unless noted
- Figures saved as PDF to `figures/`
- Processed data saved as CSV to `processed/`

---

## Data

**Raw file:** `data/BsweepTrialData.txt`  
Whitespace-delimited, `%`-prefixed comment lines at the top (skip with `comment="%"`).

**Columns (in order):**

| Column | Name | Unit |
|--------|------|------|
| 1 | r | mm |
| 2 | z | mm |
| 3 | B0_param | T (parametric sweep label) |
| 4 | Jr | A/m² |
| 5 | v | m/s (azimuthal velocity) |
| 6 | B0_dup | T (redundant copy of col 3, drop it) |

**Export details:**
- All B0 values are in one file, concatenated
- z is fixed at 0.5 mm (midplane of the h=1 mm domain)
- r is unstructured (mesh node positions), needs sorting and interpolation
- B0 sweep range: 0.5 T to 14 T (exact values — inspect with `df["B0_param"].unique()`)

**Loading:**
```python
import pandas as pd

df = pd.read_csv("data/BsweepTrialData.txt", comment="%", sep=r"\s+",
                 names=["r", "z", "B0_param", "Jr", "v", "B0_dup"])
df = df.drop(columns=["B0_dup"])
df = df.sort_values(["B0_param", "r"]).reset_index(drop=True)
```

---

## Geometry and Material Parameterspy

```python
r_in  = 0.1e-3   # m, inner electrode radius
r_out = 4.5e-3   # m, outer electrode radius
h     = 1.0e-3   # m, fluid height
L     = r_out / r_in  # = 45

sigma = 3.7e6    # S/m, gallium electrical conductivity
mu    = 1.95e-3  # Pa·s, gallium dynamic viscosity
rho   = 6095     # kg/m³, gallium density
```

Note: r and z in the data file are in **mm**. Convert to metres when computing physical
quantities (Hartmann number, diffusion length).

---

## Key Physical Quantities

**Hartmann layer thickness** (diffusion length):
```
D(B) = sqrt(mu / (sigma * B**2))   [metres]
```
At B=1T: D ≈ 23 µm. At B=10T: D ≈ 7 µm.

**Hartmann number:**
```
Ha(B) = B * h * sqrt(sigma / mu)
```
At B=1T: Ha ≈ 21.5.

**Bulk radial current density** (from current conservation, 1/r form):
```
Jr_bulk(r) = I / (2 * pi * r * h)
```
In normalised units Jr·r should be constant in the bulk. The bulk constant can be estimated
from the median of `Jr * r` at r > 3 * r_in (in mm, so `r > 0.3` mm).

---

## Analysis Tasks

### 1. Bulk validation
- For each B0, confirm Jr·r → constant as r → r_out
- The constant should be independent of B0 (it is set by total current I, not B)
- Plot Jr·r vs r for all B0 on one axes to verify

### 2. Fall-off characterisation (primary task)
The region of interest is near r_in, where Jr is elevated above its bulk 1/r value.
We care about the region where the excess has not yet decayed — specifically from
the peak down to where it reaches ~1/2 to ~2/3 of peak excess. Beyond that the
profile is bulk-like and uninteresting.

**Procedure per B0 slice:**
```
excess(r) = Jr(r) * r  -  bulk_const
```
- `peak_excess` = max of excess near r_in (take r < 3*r_in window)
- `r_half` = first r where excess drops to 0.5 * peak_excess
- `r_third` = first r where excess drops to 0.33 * peak_excess
- Region of interest mask: `excess > 0.05 * peak_excess`

Interpolate onto a regular r grid (500 points) within the region of interest before
plotting or fitting. Use `np.interp` — data is sorted by r so this is safe.

### 3. Scaling plot
- Collect r_half (or r_third) vs B0 for all sweep values
- Compute predicted scale: `delta = h / sqrt(Ha)` = `sqrt(mu * h**2 / (sigma * B**2 * h**2))` 
  = `sqrt(mu / (sigma * B**2))` = D (the Hartmann layer thickness)
  OR alternatively `delta ~ sqrt(D * h)`
- Plot r_half vs B0 on log-log axes, overlay both scalings, determine which fits

---

## Analytic Reference Solutions

### Bulk azimuthal velocity (Part 1, Sec. V confinement)
The full z-dependent solution with no-slip top/bottom:
```
U_phi(z) = (dV/dr / B) * [1 - cosh((z - h/2) / D) / cosh(h / (2D))]
```
Velocity scale: `U0 = I * B * r_in / (2 * pi * mu * h)`

### Infinite cylinder radial profile (Part 1, Eq. 39)
```python
def f_analytic(r, r_in, r_out):
    L = r_out / r_in
    rn = r / r_in  # normalised
    term1 = 0.5 * (L**2 * np.log(L) / (L**2 - 1)) * (rn/r_in - r_in/r)  # check signs
    term2 = -0.5 * (r / r_in) * np.log(r / r_in)
    return term1 + term2
```
BCs: f(1) = f(L) = 0. See Part 1 Eq. 38–39 for exact form.

---

## Figure Conventions

- All figures: `figsize=(7, 4)`, saved to `figures/` as PDF
- Label axes with units explicitly
- B0 sweep: use a sequential colormap (e.g. `plt.cm.viridis`) rather than discrete legend
  when plotting all B0 values together
- r axis in µm for near-electrode plots (multiply mm values by 1000)
- Annotate r_half on fall-off plots with a vertical dashed line

---

## File Naming

```
figures/falloff_profiles.pdf     — normalised excess Jr vs r, all B0
figures/bulk_validation.pdf      — Jr*r vs r, all B0
figures/scaling_r_half.pdf       — r_half vs B0 log-log with predicted scalings
processed/falloff_data.csv       — r_half, r_third, bulk_const per B0
```
