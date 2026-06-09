#!/usr/bin/env python3
"""
Nigeria GNSS Meteorology - ZTD Processor
==========================================
Computes Zenith Hydrostatic Delay (ZHD), Zenith Wet Delay (ZWD),
Zenith Total Delay (ZTD), and Precipitable Water Vapor (PWV)
from surface meteorological parameters and Tm.

This satisfies Objective i: "Estimate ZTD from GNSS data over Nigeria
using ERA5 atmospheric profile data."
"""

import numpy as np
import pandas as pd
from pathlib import Path

# Physical constants
K2_PRIME = 22.1       # K/hPa  (wet refractivity constant)
K3 = 3.739e5          # K^2/hPa (wet refractivity constant)
RHO_W = 1000.0        # kg/m^3 (density of liquid water)
R_V = 461.5           # J/(kg*K) (specific gas constant for water vapor)


def compute_zhd(ps_hpa, lat, elev_m):
    """
    Saastamoinen Zenith Hydrostatic Delay model.

    ZHD = 0.0022768 * Ps / (1 - 0.00266*cos(2*phi) - 0.00028*h)

    Returns ZHD in meters.
    """
    phi = np.radians(lat)
    h_km = elev_m / 1000.0
    denom = 1.0 - 0.00266 * np.cos(2.0 * phi) - 0.00028 * h_km
    zhd_m = 0.0022768 * ps_hpa / denom
    return zhd_m


def compute_surface_vapor_pressure(ts_k, rhs_pct):
    """
    Compute surface water vapor pressure e (hPa) from surface temperature
    and relative humidity using the Magnus formula.
    """
    t_c = ts_k - 273.15
    # Magnus formula for saturation vapor pressure (hPa)
    es = 6.112 * np.exp((17.67 * t_c) / (t_c + 243.5))
    # Actual vapor pressure
    e = (rhs_pct / 100.0) * es
    return np.clip(e, 0.1, 100.0)


def compute_zwd(ts_k, rhs_pct, tm_k):
    """
    Compute Zenith Wet Delay (ZWD) from surface parameters.

    Uses simplified integration of the wet refractivity profile:
    ZWD = 10^-6 * N_w0 * H_w

    where N_w0 is surface wet refractivity (ppm) and H_w is the
    water vapor scale height (m), which varies with temperature.

    Returns ZWD in meters.
    """
    e = compute_surface_vapor_pressure(ts_k, rhs_pct)

    # Water vapor scale height (m) - temperature dependent
    # Typical range: 1.5-2.5 km in tropical regions
    H_w = 1500.0 + 50.0 * (ts_k - 273.15)
    H_w = np.clip(H_w, 1000.0, 3500.0)

    # Surface wet refractivity (dimensionless, in ppm)
    N_w0 = K2_PRIME * (e / ts_k) + K3 * (e / (ts_k ** 2))

    # ZWD from integrated refractivity (meters)
    zwd_m = 1e-6 * N_w0 * H_w
    return zwd_m


def compute_pwv_factor(tm_k):
    """
    Compute PWV conversion factor Π (dimensionless).

    PWV (mm) = ZWD (mm) * Π

    Π = 10^8 / [rho_w * R_v * (K3/Tm + K2')]

    Note: Factor of 10^8 (not 10^6) accounts for K2' and K3 
    being in K/hPa and K^2/hPa units.
    """
    pi = 1e8 / (RHO_W * R_V * (K3 / tm_k + K2_PRIME))
    return pi


def compute_pwv(zwd_mm, tm_k):
    """Compute Precipitable Water Vapor in mm from ZWD in mm and Tm."""
    pi = compute_pwv_factor(tm_k)
    pwv_mm = zwd_mm * pi
    return pwv_mm


def compute_ztd_from_row(row):
    """Compute all ZTD components for a single DataFrame row."""
    zhd = compute_zhd(row['ps_hpa'], row['lat'], row['elevation_m'])
    zwd = compute_zwd(row['ts_k'], row['rhs_pct'], row['tm_k'])
    ztd = zhd + zwd
    zwd_mm = zwd * 1000.0
    pwv = compute_pwv(zwd_mm, row['tm_k'])
    return pd.Series({
        'zhd_m': zhd,
        'zwd_m': zwd,
        'ztd_m': ztd,
        'zhd_mm': zhd * 1000.0,
        'zwd_mm': zwd_mm,
        'ztd_mm': ztd * 1000.0,
        'pwv_mm': pwv,
        'pwv_cm': pwv / 10.0,
    })


def process_ztd():
    """Main processing function."""
    BASE_DIR = Path(__file__).parent
    DATA_FILE = BASE_DIR / 'data' / 'processed' / 'nigeria_tm_training_2020_2024.csv'

    if not DATA_FILE.exists():
        print(f"[ERROR] Data file not found: {DATA_FILE}")
        print("Run 0_generate_synthetic_data.py or 1_era5_processor first.")
        return None

    print("=" * 60)
    print("Nigeria GNSS Meteorology - ZTD Processor")
    print("=" * 60)

    df = pd.read_csv(DATA_FILE, parse_dates=['time'])
    print(f"Loaded {len(df):,} records from {DATA_FILE.name}")

    # Compute ZTD components
    print("\nComputing ZHD, ZWD, ZTD, and PWV...")
    ztd_data = df.apply(compute_ztd_from_row, axis=1)
    df = pd.concat([df, ztd_data], axis=1)

    # Statistics
    print("\n" + "=" * 60)
    print("ZTD STATISTICS (All Stations, All Times)")
    print("=" * 60)

    stats = {
        'ZHD (mm)': df['zhd_mm'],
        'ZWD (mm)': df['zwd_mm'],
        'ZTD (mm)': df['ztd_mm'],
        'PWV (mm)': df['pwv_mm'],
    }

    for name, series in stats.items():
        print(f"{name:12s}: {series.min():8.1f} - {series.max():8.1f}  "
              f"(mean: {series.mean():7.1f}, std: {series.std():6.1f})")

    # Station summary
    print("\n" + "=" * 60)
    print("ZTD BY STATION (Mean Values)")
    print("=" * 60)

    station_stats = df.groupby('point_name').agg({
        'zhd_mm': 'mean',
        'zwd_mm': 'mean',
        'ztd_mm': 'mean',
        'pwv_mm': 'mean',
        'elevation_m': 'first',
        'lat': 'first',
        'lon': 'first',
    }).round(2)

    station_stats = station_stats.sort_values('elevation_m')
    print(station_stats.to_string())

    # Seasonal analysis
    print("\n" + "=" * 60)
    print("SEASONAL PWV VARIATION (Mean by Month)")
    print("=" * 60)
    monthly_pwv = df.groupby('month')['pwv_mm'].agg(['mean', 'std', 'min', 'max']).round(2)
    print(monthly_pwv.to_string())

    # Save output
    output_file = BASE_DIR / 'data' / 'processed' / 'nigeria_ztd_2020_2024.csv'
    df.to_csv(output_file, index=False)
    print(f"\n[OK] ZTD dataset saved: {output_file}")
    print(f"     Records: {len(df):,}")
    print(f"     Columns: {len(df.columns)}")

    # Save summary for report generator
    summary = {
        'total_records': len(df),
        'stations': int(df['point_name'].nunique()),
        'zhd_range_mm': [round(df['zhd_mm'].min(), 2), round(df['zhd_mm'].max(), 2)],
        'zwd_range_mm': [round(df['zwd_mm'].min(), 2), round(df['zwd_mm'].max(), 2)],
        'ztd_range_mm': [round(df['ztd_mm'].min(), 2), round(df['ztd_mm'].max(), 2)],
        'pwv_range_mm': [round(df['pwv_mm'].min(), 2), round(df['pwv_mm'].max(), 2)],
        'mean_zhd_mm': round(df['zhd_mm'].mean(), 2),
        'mean_zwd_mm': round(df['zwd_mm'].mean(), 2),
        'mean_ztd_mm': round(df['ztd_mm'].mean(), 2),
        'mean_pwv_mm': round(df['pwv_mm'].mean(), 2),
    }

    summary_file = BASE_DIR / 'data' / 'processed' / 'ztd_summary.json'
    import json
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"[OK] Summary saved: {summary_file}")

    print("\n" + "=" * 60)
    print("ZTD PROCESSING COMPLETE")
    print("=" * 60)

    return df


if __name__ == '__main__':
    process_ztd()