"""
scaling.py — MHD Corbino Disk: fall-off length scaling with B₀
Reads processed/falloff_data.csv produced by analysis.py.

Produces figures/scaling_r_half.pdf  — two-panel figure:
  Left : δ vs B₀ on linear axes with fitted curve and explicit formula
  Right: same data on log-log axes showing power-law linearity and
         the two candidate theoretical scalings
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.optimize import curve_fit
import os

os.makedirs("figures", exist_ok=True)

# ── Material & geometry parameters ───────────────────────────────────────────
r_in  = 0.1e-3   # m
h     = 1.0e-3   # m
sigma = 3.7e6    # S/m
mu    = 1.95e-3  # Pa·s

# ── Load processed data ───────────────────────────────────────────────────────
df = pd.read_csv("processed/falloff_data.csv")

B0    = df["B0"].values                                # T
# Fall-off length = distance from inner electrode to half-excess point
delta = (df["r_half_mm"].values - r_in * 1e3) * 1e-3  # m
delta_third = (df["r_third_mm"].values - r_in * 1e3) * 1e-3

# ── Physical scaling predictions ─────────────────────────────────────────────
# D(B)   = sqrt(mu / (sigma * B^2))          Hartmann layer thickness  ~ B^-1
# sqrtDh = sqrt(D * h)  = (mu*h/sigma)^1/4  * B^-1/2                  ~ B^-0.5
D_arr    = np.sqrt(mu / (sigma * B0**2))
sqDh_arr = np.sqrt(D_arr * h)

# ── Curve fit: δ = A · B^n ────────────────────────────────────────────────────
def power_law(B, A, n):
    return A * B**n

popt, pcov = curve_fit(power_law, B0, delta, p0=[200e-6, -0.5])
A_fit, n_fit = popt
A_err, n_err = np.sqrt(np.diag(pcov))

# Amplitude of the physical prediction  δ = α · sqrt(D·h)
# median of δ / sqrt(D·h) across all B0
alpha_sqDh  = float(np.median(delta / sqDh_arr))
alpha_D     = float(np.median(delta / D_arr))

print("=" * 60)
print("SCALING ANALYSIS RESULTS")
print("=" * 60)
print(f"\nPower-law fit:  δ = A · B^n")
print(f"  A = {A_fit*1e6:.1f} ± {A_err*1e6:.1f}  µm·T^(-n)")
print(f"  n = {n_fit:.3f} ± {n_err:.3f}")
print(f"\nPhysical scaling amplitudes:")
print(f"  δ / D(B)       = {alpha_D:.2f}    (D ~ B^-1)")
print(f"  δ / √(D·h)     = {alpha_sqDh:.3f}   (√(D·h) ~ B^-0.5)")
print(f"\n{'B0 (T)':>8}  {'δ_half (µm)':>12}  {'fit (µm)':>10}  "
      f"{'1.41√Dh (µm)':>14}  {'D (µm)':>10}")
print("-" * 60)
for i in range(len(B0)):
    d_fit   = power_law(B0[i], A_fit, n_fit)
    d_th    = alpha_sqDh * sqDh_arr[i]
    print(f"{B0[i]:>8.1f}  {delta[i]*1e6:>12.1f}  {d_fit*1e6:>10.1f}  "
          f"{d_th*1e6:>14.1f}  {D_arr[i]*1e6:>10.2f}")

# ── Smooth curves for plotting ────────────────────────────────────────────────
B_fine   = np.linspace(B0.min(), B0.max(), 400)
B_fine_l = np.logspace(np.log10(B0.min()), np.log10(B0.max()), 400)

fit_fine  = power_law(B_fine,   A_fit, n_fit)
fit_fine_l = power_law(B_fine_l, A_fit, n_fit)

D_fine    = np.sqrt(mu / (sigma * B_fine_l**2))
sqDh_fine = np.sqrt(D_fine * h)
th_fine   = alpha_sqDh * sqDh_fine
D_curve   = alpha_D    * D_fine

# ── Labels for the explicit formula ──────────────────────────────────────────
fit_label_lin = (rf"Fit:  $\delta_{{1/2}} = {A_fit*1e6:.0f}\,\mu\mathrm{{m}}"
                 rf"\cdot B_0^{{{n_fit:.2f}}}$")
th_label      = (rf"$\alpha\sqrt{{Dh}}$  $(\alpha={alpha_sqDh:.2f},\;B^{{-1/2}})$")
D_label       = (rf"$\alpha D$  $(\alpha={alpha_D:.1f},\;B^{{-1}})$")
fit_label_log = (rf"Fit: $B_0^{{{n_fit:.2f}}}$")

# ── Figure: two panels ────────────────────────────────────────────────────────
fig, (ax_lin, ax_log) = plt.subplots(1, 2, figsize=(13, 4.5))

# ── Left panel: linear axes ───────────────────────────────────────────────────
ax_lin.scatter(B0, delta * 1e6, color="royalblue", s=30, zorder=5,
               label=r"$\delta_{1/2}$  (data)")
ax_lin.scatter(B0, delta_third * 1e6, color="cornflowerblue", s=18,
               marker="s", alpha=0.65, zorder=4,
               label=r"$\delta_{1/3}$  (data)")
ax_lin.plot(B_fine, fit_fine * 1e6, "k-",  lw=2.0, zorder=6, label=fit_label_lin)

D_fine_lin    = np.sqrt(mu / (sigma * B_fine**2))
sqDh_fine_lin = np.sqrt(D_fine_lin * h)
ax_lin.plot(B_fine, alpha_sqDh * sqDh_fine_lin * 1e6,
            "r--", lw=1.8, label=th_label)

ax_lin.set_xlabel("$B_0$  (T)", fontsize=12)
ax_lin.set_ylabel(r"Fall-off length  $\delta_{1/2} = r_{1/2} - r_\mathrm{in}$  (µm)",
                  fontsize=11)
ax_lin.set_title("Fall-off length vs applied field\n(linear axes)", fontsize=10)
ax_lin.legend(fontsize=8.5, loc="upper right")
ax_lin.grid(ls=":", alpha=0.4)
ax_lin.set_xlim(left=0)
ax_lin.set_ylim(bottom=0)

# Annotate the fitted formula prominently
ax_lin.annotate(
    fit_label_lin.replace("Fit:  ", ""),
    xy=(7.5, power_law(7.5, A_fit, n_fit) * 1e6),
    xytext=(5.5, 195),
    fontsize=9.5,
    color="black",
    arrowprops=dict(arrowstyle="->", color="black", lw=0.8),
)

# ── Right panel: log-log axes ────────────────────────────────────────────────
ax_log.loglog(B0, delta * 1e6,       "o", ms=5,  color="royalblue",
              label=r"$\delta_{1/2}$  (data)", zorder=5)
ax_log.loglog(B0, delta_third * 1e6, "s", ms=4,  color="cornflowerblue",
              alpha=0.65, label=r"$\delta_{1/3}$  (data)", zorder=4)
ax_log.loglog(B_fine_l, fit_fine_l  * 1e6, "k-",  lw=2.0, label=fit_label_log)
ax_log.loglog(B_fine_l, th_fine     * 1e6, "r--", lw=1.8, label=th_label)
ax_log.loglog(B_fine_l, D_curve     * 1e6, color="darkorange",
              ls="-.", lw=1.8, label=D_label)

# Reference slope annotations
x_ann = 8.0
ax_log.annotate("", xy=(x_ann * 1.6, power_law(x_ann * 1.6, A_fit, n_fit) * 1e6),
                xytext=(x_ann, power_law(x_ann, A_fit, n_fit) * 1e6),
                arrowprops=dict(arrowstyle="-", color="k", lw=0.6))

ax_log.set_xlabel("$B_0$  (T)", fontsize=12)
ax_log.set_ylabel(r"$\delta_{1/2}$  (µm)", fontsize=11)
ax_log.set_title("Same data on log-log axes\n"
                 "(straight line confirms power law)", fontsize=10)
ax_log.xaxis.set_major_formatter(ticker.ScalarFormatter())
ax_log.yaxis.set_major_formatter(ticker.ScalarFormatter())
ax_log.legend(fontsize=8.5, loc="lower left")
ax_log.grid(True, which="both", ls=":", alpha=0.35)

fig.suptitle(
    rf"$\delta_{{1/2}} = {A_fit*1e6:.0f}\,\mu\mathrm{{m}}\cdot B_0^{{{n_fit:.2f}}}$"
    rf"  $\approx$  $\alpha\,\sqrt{{Dh}}$   with  $\alpha = {alpha_sqDh:.2f}$",
    fontsize=12, y=1.01,
)
fig.tight_layout()
fig.savefig("figures/scaling_r_half.pdf", bbox_inches="tight")
plt.close(fig)
print("\nSaved: figures/scaling_r_half.pdf")
