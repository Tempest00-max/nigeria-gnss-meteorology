#!/usr/bin/env python3
"""
1_era5_production_2017_2022.py — FINAL FIX
============================================
Uses correct ARCO-ERA5 zarr-v3 path with PROPER variable names.
The zarr store has variables named 'temperature', 'relative_humidity', 'geopotential'
NOT 't', 'r', 'z'.
"""

import os
import sys
import warnings
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

NIGERIA_STATIONS = [
    # North-West (7)
    (11.99, 8.53, "Kano", 480), (12.00, 8.08, "Katsina", 520),
    (10.52, 7.43, "Kaduna", 610), (12.74, 10.97, "Dutse", 450),
    (11.07, 7.72, "Birnin_Kebbi", 230), (12.45, 4.20, "Sokoto", 300),
    (12.00, 6.78, "Gusau", 460),
    # North-East (6)
    (11.75, 13.15, "Maiduguri", 300), (11.85, 13.15, "Damaturu", 360),
    (10.30, 9.75, "Bauchi", 590), (10.28, 11.17, "Gombe", 420),
    (10.60, 12.18, "Yola", 190), (11.08, 12.68, "Jalingo", 240),
    # North-Central + FCT (7)
    (9.08, 7.40, "Abuja", 360), (9.08, 5.12, "Minna", 260),
    (8.12, 9.68, "Lafia", 180), (7.72, 8.52, "Makurdi", 100),
    (7.80, 6.73, "Lokoja", 40), (8.68, 4.58, "Ilorin", 310),
    (9.93, 8.88, "Jos", 1280),
    # South-West (6)
    (6.45, 3.40, "Lagos", 15), (7.15, 3.35, "Abeokuta", 60),
    (7.38, 3.93, "Ibadan", 120), (7.60, 5.22, "Ado_Ekiti", 450),
    (7.80, 4.58, "Akure", 350), (7.78, 4.55, "Osogbo", 300),
    # South-East (5)
    (6.02, 6.78, "Enugu", 180), (6.18, 6.73, "Owerri", 70),
    (5.38, 7.00, "Umuahia", 130), (5.90, 7.38, "Abakaliki", 390),
    (6.02, 7.50, "Awka", 100),
    # South-South (6)
    (5.02, 7.93, "Uyo", 12), (4.98, 8.35, "Calabar", 35),
    (4.77, 7.02, "Yenagoa", 6), (5.53, 6.02, "Port_Harcourt", 20),
    (6.33, 5.60, "Asaba", 18), (7.25, 5.20, "Benin_City", 85),
]

YEARS = [2017, 2018, 2019, 2020, 2021, 2022]
TIME_STEPS = [0, 6, 12, 18]

BASE_DIR = Path(__file__).parent
PROCESSED_DIR = BASE_DIR / 'data' / 'processed'
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

GCS_ARCO = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"

# =============================================================================
# PHYSICAL FORMULAS
# =============================================================================

def compute_rh_from_dewpoint(t_k, td_k):
    t = t_k - 273.15
    td = td_k - 273.15
    a, b = 17.625, 243.04
    rh = 100 * np.exp((a * td / (b + td)) - (a * t / (b + t)))
    return float(np.clip(rh, 0.1, 100.0))


def compute_water_vapor_pressure(rh, t_celsius):
    rh = np.clip(rh, 0.1, 100.0)
    es = 6.112 * np.exp((17.67 * t_celsius) / (t_celsius + 243.5))
    return (rh / 100.0) * es


def compute_tm_from_profile(temperature_k, relative_humidity, geopotential_height_m):
    temp = np.asarray(temperature_k, dtype=np.float64)
    rh = np.asarray(relative_humidity, dtype=np.float64)
    h = np.asarray(geopotential_height_m, dtype=np.float64)

    sort_idx = np.argsort(h)
    temp = temp[sort_idx]; rh = rh[sort_idx]; h = h[sort_idx]

    valid = ~(np.isnan(temp) | np.isnan(rh) | np.isnan(h))
    if valid.sum() < 3:
        return np.nan

    temp = temp[valid]; rh = rh[valid]; h = h[valid]
    t_c = temp - 273.15
    e = compute_water_vapor_pressure(rh, t_c)

    n = len(h)
    if n < 2:
        return np.nan

    t_layer = 0.5 * (temp[:-1] + temp[1:])
    e_layer = 0.5 * (e[:-1] + e[1:])
    dh_km = np.diff(h) / 1000.0

    numerator = np.nansum((e_layer / t_layer) * dh_km)
    denominator = np.nansum((e_layer / (t_layer ** 2)) * dh_km)

    if denominator == 0 or np.isnan(denominator):
        return np.nan

    tm = numerator / denominator
    if tm < 200 or tm > 320:
        return np.nan
    return float(tm)


def geopotential_to_height(z, latitude=10.0):
    phi = np.radians(latitude)
    g = 9.80620 * (1 - 2.6442e-3 * np.cos(2*phi) + 5.8e-6 * (np.cos(2*phi)**2))
    return z / g


# =============================================================================
# MAIN EXTRACTION
# =============================================================================

def extract_era5_arco():
    """Extract ERA5 data from Google Cloud ARCO for 2017-2022."""
    print("=" * 70)
    print("NIGERIA GNSS METEOROLOGY — PRODUCTION ERA5 PIPELINE")
    print("Source: Google Cloud ARCO-ERA5 (zarr-v3)")
    print("Period: 2017-2022 | Stations: 37 | Resolution: 6-hourly")
    print("=" * 70)

    print("\n[1/4] Connecting to Google Cloud ARCO...")
    print(f"      Path: {GCS_ARCO}")

    try:
        import xarray as xr
    except ImportError:
        print("      ✗ xarray not installed: pip install xarray")
        return []

    try:
        print("      Opening with anonymous access...")
        ds = xr.open_zarr(
            GCS_ARCO,
            chunks=None,
            storage_options=dict(token='anon'),
        )

        # Filter to valid time range
        if 'valid_time_start' in ds.attrs and 'valid_time_stop' in ds.attrs:
            ds = ds.sel(time=slice(ds.attrs['valid_time_start'], ds.attrs['valid_time_stop']))

        print(f"      ✓ Dataset loaded: {dict(ds.dims)}")
        print(f"      ✓ Variables: {list(ds.data_vars)[:15]}")

    except Exception as e:
        print(f"      ✗ Failed: {str(e)[:150]}")
        return []

    # Filter spatial
    print("\n[2/4] Subsetting spatial domain...")
    try:
        ds = ds.sel(latitude=slice(14.5, 3.5), longitude=slice(2.0, 15.0))
        print(f"      ✓ Spatial subset OK")
    except Exception as e:
        print(f"      ✗ {e}")
        return []

    # Filter temporal
    print("[3/4] Filtering temporal domain (2017-2022)...")
    try:
        ds = ds.sel(time=ds.time.dt.year.isin(YEARS))
        print(f"      ✓ Time steps: {ds.sizes.get('time', '?')}")
    except Exception as e:
        print(f"      ✗ {e}")
        return []

    # CRITICAL FIX: Check for pressure-level variables with CORRECT names
    available = list(ds.data_vars)
    print(f"      Available vars: {available[:20]}...")

    # ARCO uses FULL names, not ECMWF short names
    temp_var = 'temperature' if 'temperature' in available else None
    rh_var = 'relative_humidity' if 'relative_humidity' in available else None
    z_var = 'geopotential' if 'geopotential' in available else None

    # Surface variables
    ts_var = '2m_temperature' if '2m_temperature' in available else None
    td_var = '2m_dewpoint_temperature' if '2m_dewpoint_temperature' in available else None
    sp_var = 'surface_pressure' if 'surface_pressure' in available else None

    # Check if we have what we need
    if not all([temp_var, rh_var, z_var]):
        print("      ✗ Missing pressure-level variables!")
        print(f"      Looking for: temperature, relative_humidity, geopotential")
        print(f"      Found: {available[:30]}")

        # DEBUG: Show ALL variables
        print(f"\n      ALL VARIABLES IN STORE:")
        for i, v in enumerate(sorted(available)):
            print(f"        {i+1:3d}. {v}")
        return []

    print(f"      ✓ Pressure: temperature={temp_var}, RH={rh_var}, geopotential={z_var}")
    print(f"      ✓ Surface: 2m_temp={ts_var}, 2m_dewpoint={td_var}, sp={sp_var}")

    # Extract stations
    print("\n[4/4] Extracting 37 stations...")
    all_records = []
    total_stations = len(NIGERIA_STATIONS)

    for idx, (lat, lon, name, elev) in enumerate(NIGERIA_STATIONS, 1):
        print(f"      [{idx:2d}/{total_stations}] {name}...", end=" ")

        try:
            point = ds.sel(latitude=lat, longitude=lon, method='nearest')
            times = pd.to_datetime(point['time'].values)

            station_records = []
            for i, t in enumerate(times):
                if t.hour not in TIME_STEPS:
                    continue

                try:
                    # Extract pressure-level profiles
                    temp_prof = point[temp_var].isel(time=i).values
                    rh_prof = point[rh_var].isel(time=i).values
                    z_prof = point[z_var].isel(time=i).values
                except Exception as e:
                    continue

                temp_prof = np.asarray(temp_prof).flatten()
                rh_prof = np.asarray(rh_prof).flatten()
                z_prof = np.asarray(z_prof).flatten()
                h_prof = geopotential_to_height(z_prof, lat)

                tm = compute_tm_from_profile(temp_prof, rh_prof, h_prof)

                # Surface variables
                try:
                    ts = point[ts_var].isel(time=i).values if ts_var else np.nan
                except:
                    ts = np.nan
                try:
                    ps = point[sp_var].isel(time=i).values if sp_var else np.nan
                except:
                    ps = np.nan
                try:
                    td = point[td_var].isel(time=i).values if td_var else np.nan
                except:
                    td = np.nan

                # Compute RH
                if not np.isnan(ts) and not np.isnan(td):
                    rhs = compute_rh_from_dewpoint(float(ts), float(td))
                elif len(rh_prof) > 0:
                    rhs = float(rh_prof[0])
                else:
                    rhs = np.nan

                ts_k = float(ts) if not np.isnan(ts) else np.nan
                ps_hpa = float(ps) / 100.0 if not np.isnan(ps) else np.nan

                if not np.isnan(tm) and not np.isnan(ts_k):
                    station_records.append({
                        'time': t, 'lat': lat, 'lon': lon, 'point_name': name,
                        'tm_k': round(float(tm), 3), 'ts_k': round(ts_k, 3),
                        'ps_hpa': round(ps_hpa, 2) if not np.isnan(ps_hpa) else np.nan,
                        'rhs_pct': round(rhs, 2) if not np.isnan(rhs) else np.nan,
                        'elevation_m': elev, 'year': t.year, 'month': t.month,
                        'day': t.day, 'hour': t.hour, 'doy': t.dayofyear,
                    })

            all_records.extend(station_records)
            print(f"✓ {len(station_records)} records")

        except Exception as e:
            print(f"✗ {str(e)[:60]}")
            continue

    ds.close()
    return all_records


def main():
    records = extract_era5_arco()

    if not records:
        print("\n[ERROR] No records extracted from GCS.")
        print("        Run with synthetic fallback: python 0_generate_synthetic_data.py")
        return

    df = pd.DataFrame(records)
    df = df.sort_values(['time', 'point_name']).reset_index(drop=True)

    output_file = PROCESSED_DIR / 'nigeria_tm_training_2017_2022.csv'
    df.to_csv(output_file, index=False)

    meta = {
        "source": "GCS_ARCO_ERA5_zarr-v3",
        "path": GCS_ARCO,
        "generated": datetime.now().isoformat(),
        "records": len(df),
        "stations": int(df['point_name'].nunique()),
        "years": sorted(df['year'].unique().tolist()),
        "time_range": f"{df['time'].min()} to {df['time'].max()}",
        "tm_range": [round(df['tm_k'].min(), 2), round(df['tm_k'].max(), 2)],
    }
    meta_file = PROCESSED_DIR / 'data_source_meta.json'
    with open(meta_file, 'w') as f:
        json.dump(meta, f, indent=2)

    print("\n" + "=" * 70)
    print("✓ REAL ERA5 DATA EXTRACTION COMPLETE")
    print("=" * 70)
    print(f"Total records:    {len(df):,}")
    print(f"Stations:         {df['point_name'].nunique()}")
    print(f"Time range:       {df['time'].min()} to {df['time'].max()}")
    print(f"Years:            {df['year'].min()}–{df['year'].max()}")
    print(f"Tm range:         {df['tm_k'].min():.2f} K – {df['tm_k'].max():.2f} K")
    print(f"Saved:            {output_file}")
    print("=" * 70)

    print("\n⚡ Next steps:")
    print("   python validate_production_data.py")
    print("   python 2_model_fitter_ML.py")
    print("   python 3_ztd_processor.py")
    print("   python report_generator_v2.py")
    print("   python backend_v3.py")


if __name__ == '__main__':
    main()