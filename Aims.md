# Aims.md — Physical Context and Scientific Goals

This document gives the physical and theoretical background for the post-processing scripts.
Read this to understand *why* the analysis is structured the way it is.

---

## Physical Setup

A Corbino disk geometry: liquid gallium fills an annular region between inner radius r_in and
outer radius r_out, height h. Electrodes at r_in and r_out drive a radial current I. A uniform
axial magnetic field B = B_z ẑ is applied.

The J × B body force on the radial current is azimuthal: F_phi = -J_r * B. This drives
azimuthal rotation of the fluid. The problem is to find U_phi(r, z) and J_r(r, z).

The two coupled PDEs governing this (in Cartesian/axisymmetric form, from Part 2) are:

```
(M̂ + ∂_z² - 1) u  =  -∂_r Φ           [Stokes equation, azimuthal]
∂_r u + u/r        =  (M̂ + α_ωτ ∂_z² + 1/r²) Φ   [current conservation + Ohm's law]
```

where u is the normalised azimuthal velocity, Φ is electric potential, and M̂ is the
radial part of the cylindrical Laplacian for azimuthal fields.

In COMSOL these are solved via manually coupled Electric Currents + Laminar Flow, because
the built-in MHD node does not correctly route the azimuthal Lorentz force to the swirl
momentum equation and does not feed back the back-EMF. The manual coupling uses:
- Volume Force: F_phi = -J_r * B  (in the azimuthal momentum equation)
- External Current Density: J_EMF = sigma * U_phi * B  (back-EMF, fed into EC module)

---

## The Hartmann Layer

With no-slip boundary conditions at z=0 and z=h, the velocity profile develops thin boundary
layers of thickness:

```
D = sqrt(mu / (sigma * B²))
```

This is the Hartmann layer (here called diffusion length). At B=1T for gallium: D ≈ 23 µm.
At B=10T: D ≈ 7 µm. The fluid height h=1mm, so the ratio h/D = Ha (Hartmann number).

The bulk (z-independent) velocity profile is well approximated by the infinite-cylinder
analytic solution (Part 1, Eq. 39) when h >> D, i.e. large Ha. The no-slip layers are
confined near z=0 and z=h, and the bulk is essentially 2D in r.

The COMSOL mesh must resolve D: first element size set to D/5. This spans 3 orders of
magnitude from the Hartmann layer (~µm) to the domain size (~mm). This is why the mesh
is non-uniform and why the exported r coordinates are not evenly spaced.

---

## What the Simulations Are Computing

The 2D axisymmetric COMSOL model solves for U_phi(r, z) and J_r(r, z) across the full
(r, z) domain. The export taken here is:

- **J_r at z = h/2 (midplane):** this is where the bulk 1/r behaviour should be cleanest,
  far from the Hartmann layers at z=0 and z=h.
- **v (azimuthal velocity) at z = h/2:** same reasoning.

---

## The Fall-Off Study: Scientific Question

In the bulk (far from the inner electrode), J_r = I/(2πrh), i.e. J_r * r = const. This
follows purely from current conservation (∇·J = 0) in a cylinder.

Near the inner electrode at r_in, the current enters from a finite-height contact. The
current distribution is not purely radial there — it must redistribute from the contact
geometry into the bulk 1/r profile. This redistribution happens over a characteristic
length scale δ in the r direction.

**The scientific question:** how does δ scale with B and h?

**Proposed scaling:** δ ~ h / sqrt(Ha) = sqrt(mu * h² / (sigma * B² * h²)) = D

That is, the fall-off length is the Hartmann layer thickness itself. Alternatively, it
could scale as sqrt(D * h) (geometric mean of the two length scales).

The sweep over B (at fixed h=1mm) tests the B-dependence. A previous sweep over h (at
fixed B) was planned (h = [0.2, 0.5, 1, 2, 5] mm) but is not in this dataset.

**What to measure:** r_half = the radial distance from r_in at which the excess
(J_r * r - bulk_const) has decayed to 50% of its peak value near the electrode.
Plot r_half vs B on log-log axes. The slope gives the power law exponent.

If δ ~ D ~ B^(-1): slope = -1 on log-log plot.
If δ ~ sqrt(D*h) ~ B^(-1/2): slope = -0.5.

---

## Relationship to COMSOL Setup

The COMSOL model is: 2D axisymmetric, Electric Currents + Laminar Flow (swirl flow),
manual coupling via Volume Force and External Current Density. Segregated solver.

The inner electrode (r = r_in) has a specified voltage or current. The outer electrode
(r = r_out) is grounded. Top and bottom boundaries (z = 0, z = h) are no-slip walls,
electrically insulating (n·J = 0).

The export was taken along a horizontal line at z = h/2 across the full r range,
for each value of B in the parametric sweep.

---

## References

- Part 1: Keser, "Metal Hydro" (CSIRO, Feb 2026) — derives the governing equations,
  infinite-cylinder analytic solution (Eq. 39), confinement in z (Sec. V).
- Part 2: Keser, "Liquid Metal Boundary and Interface Conditions" (Bilkent, Feb 2026) —
  derives the coupled PDE system, bulk asymptotic solution, FEM formulation, COMSOL
  implementation details.
