"""process_2d.py — First-stage processing for the 2D B-sweep export.

This script reconstructs the nearly structured (r, z) mesh in
data_2D/BSweep2D.txt by rounding coordinates onto the exported COMSOL grid,
then extracts the half-excess fall-off radius r_1/2 for every z-slice and B0.

Outputs:
  processed/falloff_2d_by_z.csv
  processed/mesh_report_2d.txt
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data_2D" / "BSweep2D.txt"
PROCESSED_DIR = Path(__file__).resolve().parent / "processed"

R_IN_MM = 0.1
R_BULK_MM = 0.3
ROUND_DECIMALS = 5
N_INTERP = 500

SIGMA = 3.7e6
MU = 1.95e-3


def load_2d_data(data_file: Path = DATA_FILE, decimals: int = ROUND_DECIMALS) -> pd.DataFrame:
    df = pd.read_csv(data_file, comment="%", sep=r"\s+", names=["r", "z", "B0", "Jr"])
    df = df.astype({"r": float, "z": float, "B0": float, "Jr": float})
    df["r_q"] = df["r"].round(decimals)
    df["z_q"] = df["z"].round(decimals)
    return df.sort_values(["B0", "z_q", "r_q"]).reset_index(drop=True)


def hartmann_thickness_mm(B0: float) -> float:
    return float(np.sqrt(MU / (SIGMA * B0**2)) * 1e3)


def build_structured_grid(slice_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    grouped = (
        slice_df.groupby(["r_q", "z_q"], as_index=False)["Jr"]
        .mean()
        .sort_values(["z_q", "r_q"])
    )

    r_vals = np.array(sorted(grouped["r_q"].unique()), dtype=float)
    z_vals = np.array(sorted(grouped["z_q"].unique()), dtype=float)

    pivot = (
        grouped.pivot(index="r_q", columns="z_q", values="Jr")
        .reindex(index=r_vals, columns=z_vals)
    )

    if pivot.isna().any().any():
        raise ValueError(
            "Structured grid reconstruction produced missing values; "
            "the fallback interpolation path has not been implemented yet."
        )

    metadata = {
        "rows": int(len(slice_df)),
        "groups": int(len(grouped)),
        "n_r": int(len(r_vals)),
        "n_z": int(len(z_vals)),
        "duplicate_groups": int((slice_df.groupby(["r_q", "z_q"]).size() > 1).sum()),
    }
    return r_vals, z_vals, pivot.to_numpy(dtype=float), metadata


def bulk_constant(r_mm: np.ndarray, jr: np.ndarray) -> float:
    bulk_mask = r_mm > R_BULK_MM
    if bulk_mask.sum() < 3:
        bulk_mask = r_mm >= np.quantile(r_mm, 0.75)
    return float(np.median(jr[bulk_mask] * (r_mm[bulk_mask] * 1e-3)))


def first_crossing(r_mm: np.ndarray, y: np.ndarray, level: float) -> float:
    below = np.where(y <= level)[0]
    if len(below) == 0:
        return float(r_mm[-1])
    idx = int(below[0])
    if idx == 0:
        return float(r_mm[0])
    x0, x1 = float(r_mm[idx - 1]), float(r_mm[idx])
    y0, y1 = float(y[idx - 1]), float(y[idx])
    if np.isclose(y1, y0):
        return x1
    frac = (level - y0) / (y1 - y0)
    return x0 + frac * (x1 - x0)


def extract_falloff_for_profile(r_mm: np.ndarray, jr: np.ndarray) -> dict[str, float]:
    jr_r = jr * (r_mm * 1e-3)
    bulk = bulk_constant(r_mm, jr)
    excess = jr_r - bulk

    near_mask = r_mm < R_BULK_MM
    if not np.any(near_mask):
        return {
            "bulk_const": bulk,
            "peak_excess": np.nan,
            "r_peak_mm": np.nan,
            "r_half_mm": np.nan,
            "r_third_mm": np.nan,
            "delta_half_um": np.nan,
            "delta_third_um": np.nan,
            "roi_points": 0,
        }

    near_idx = np.where(near_mask)[0]
    local_idx = int(near_idx[np.argmax(excess[near_mask])])
    peak_excess = float(excess[local_idx])
    r_peak_mm = float(r_mm[local_idx])

    if not np.isfinite(peak_excess) or peak_excess <= 0:
        return {
            "bulk_const": bulk,
            "peak_excess": peak_excess,
            "r_peak_mm": r_peak_mm,
            "r_half_mm": np.nan,
            "r_third_mm": np.nan,
            "delta_half_um": np.nan,
            "delta_third_um": np.nan,
            "roi_points": 0,
        }

    decay_r = r_mm[local_idx:]
    decay_excess = excess[local_idx:]
    roi_mask = decay_excess > 0.05 * peak_excess
    if roi_mask.sum() < 2:
        return {
            "bulk_const": bulk,
            "peak_excess": peak_excess,
            "r_peak_mm": r_peak_mm,
            "r_half_mm": np.nan,
            "r_third_mm": np.nan,
            "delta_half_um": np.nan,
            "delta_third_um": np.nan,
            "roi_points": int(roi_mask.sum()),
        }

    r_roi = decay_r[roi_mask]
    excess_roi = decay_excess[roi_mask]

    r_grid = np.linspace(float(r_roi[0]), float(r_roi[-1]), N_INTERP)
    excess_grid = np.interp(r_grid, r_roi, excess_roi)

    r_half_mm = first_crossing(r_grid, excess_grid, 0.5 * peak_excess)
    r_third_mm = first_crossing(r_grid, excess_grid, 0.33 * peak_excess)

    return {
        "bulk_const": bulk,
        "peak_excess": peak_excess,
        "r_peak_mm": r_peak_mm,
        "r_half_mm": r_half_mm,
        "r_third_mm": r_third_mm,
        "delta_half_um": (r_half_mm - R_IN_MM) * 1e3,
        "delta_third_um": (r_third_mm - R_IN_MM) * 1e3,
        "roi_points": int(roi_mask.sum()),
    }


def process_slice(b0: float, slice_df: pd.DataFrame) -> tuple[list[dict[str, float]], dict[str, int]]:
    r_vals, z_vals, jr_grid, metadata = build_structured_grid(slice_df)

    records: list[dict[str, float]] = []
    for col_idx, z_mm in enumerate(z_vals):
        profile = jr_grid[:, col_idx]
        metrics = extract_falloff_for_profile(r_vals, profile)
        records.append(
            {
                "B0_T": float(b0),
                "z_mm": float(z_mm),
                **metrics,
            }
        )

    metadata.update(
        {
            "b0": float(b0),
            "hartmann_mm": hartmann_thickness_mm(float(b0)),
            "midplane_z_mm": float(z_vals[np.argmin(np.abs(z_vals - 0.5))]),
        }
    )
    return records, metadata


def write_mesh_report(report_rows: list[dict[str, int]], output_file: Path) -> None:
    lines = []
    lines.append("2D mesh reconstruction report")
    lines.append("=" * 32)
    lines.append(f"round_decimals = {ROUND_DECIMALS}")
    lines.append("")
    lines.append(f"{'B0 (T)':>6}  {'rows':>8}  {'groups':>8}  {'n_r':>5}  {'n_z':>5}  {'dup':>6}  {'D (um)':>8}")
    lines.append("-" * 56)
    for row in report_rows:
        lines.append(
            f"{row['b0']:>6.1f}  {row['rows']:>8,d}  {row['groups']:>8,d}  {row['n_r']:>5d}  "
            f"{row['n_z']:>5d}  {row['duplicate_groups']:>6d}  {row['hartmann_mm']*1e3:>8.2f}"
        )
    output_file.write_text("\n".join(lines) + "\n")


def main() -> None:
    PROCESSED_DIR.mkdir(exist_ok=True)

    df = load_2d_data()
    b0_values = sorted(df["B0"].unique())

    all_records: list[dict[str, float]] = []
    report_rows: list[dict[str, int]] = []

    print(f"Loaded {len(df):,} rows from {DATA_FILE.name}")
    print(f"Unique B0 values: {len(b0_values)}")

    for b0 in b0_values:
        slice_df = df[df["B0"] == b0].copy()
        records, metadata = process_slice(b0, slice_df)
        all_records.extend(records)
        report_rows.append(metadata)

        midplane_row = min(records, key=lambda row: abs(row["z_mm"] - 0.5))
        print(
            f"B0={b0:>4.1f} T  rows={metadata['rows']:,}  groups={metadata['groups']:,}  "
            f"n_r={metadata['n_r']:>4d}  n_z={metadata['n_z']:>3d}  "
            f"δ1/2(midplane)={midplane_row['delta_half_um']:>8.2f} µm"
        )

    results = pd.DataFrame(all_records)
    results = results.sort_values(["B0_T", "z_mm"]).reset_index(drop=True)
    results.to_csv(PROCESSED_DIR / "falloff_2d_by_z.csv", index=False)

    write_mesh_report(report_rows, PROCESSED_DIR / "mesh_report_2d.txt")

    print(f"\nSaved {len(results):,} falloff records to processed/falloff_2d_by_z.csv")
    print("Saved mesh report to processed/mesh_report_2d.txt")


if __name__ == "__main__":
    main()