#!/usr/bin/env python3
"""
1_era5_ncar_download.py — FAST ERA5 from NSF NCAR GDEX
========================================================
Downloads real ERA5 pressure-level NetCDF files via HTTP.
No CDS queue. No GCS auth issues. Direct download from US government-backed archive.

Source: NSF NCAR GDEX Dataset d633000
URL: https://gdex.ucar.edu/datasets/d633000/
Coverage: 1940-01-01 to 2026-03-31 (hourly, 0.25°)
Variables: temperature, relative_humidity, geopotential (37 pressure levels)
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import requests
import xarray as xr
from tqdm import tqdm
import warnings
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
PRESSURE_LEVELS = [1000, 950, 925, 900, 875, 850, 800, 750, 700, 600, 500, 400, 300]

BASE_DIR = Path(__file__).parent
RAW_DIR = BASE_DIR / 'data' / 'raw_era5_ncar'
PROCESSED_DIR = BASE_DIR / 'data' / 'processed'
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# NCAR GDEX BASE URL
# =============================================================================
# NCAR stores ERA5 in yearly folders with daily files
# Format: https://gdex.ucar.edu/datasets/d633000/data/YYYY/YYYYMMDD.nc
# Or via AWS S3: s3://ncar-era5/...
NCAR_BASE = "https://gdex.ucar.edu/datasets/d633000/data"

# Alternative: Use the AWS Open Data Registry for ERA5
# AWS_ERA5 = "s3://era5-pds/"  # Requires boto3

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
# DOWNLOAD FROM NCAR GDEX
# =============================================================================

def download_ncar_file(year, month, day, output_dir):
    """Download a single daily NetCDF file from NCAR GDEX."""
    date_str = f"{year}{month:02d}{day:02d}"
    url = f"{NCAR_BASE}/{year}/{date_str}.nc"
    output_file = output_dir / f"era5_{date_str}.nc"

    if output_file.exists():
        return output_file

    try:
        print(f"    Downloading {date_str}...", end=" ")
        response = requests.get(url, timeout=60, stream=True)
        if response.status_code == 200:
            with open(output_file, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"✓ ({len(response.content)//1024//1024}MB)")
            return output_file
        else:
            print(f"✗ HTTP {response.status_code}")
            return None
    except Exception as e:
        print(f"✗ {str(e)[:50]}")
        return None


def extract_from_ncar_file(nc_file, stations):
    """Extract station data from a NetCDF file."""
    records = []

    try:
        ds = xr.open_dataset(nc_file)
    except Exception as e:
        print(f"    [ERR] Cannot open {nc_file}: {e}")
        return records

    # Check available variables
    available = list(ds.data_vars)

    # NCAR uses ECMWF short names
    temp_var = 't' if 't' in available else None
    rh_var = 'r' if 'r' in available else None
    z_var = 'z' if 'z' in available else None
    ts_var = 't2m' if 't2m' in available else None
    td_var = 'd2m' if 'd2m' in available else None
    sp_var = 'sp' if 'sp' in available else None

    if not all([temp_var, rh_var, z_var]):
        print(f"    [ERR] Missing pressure vars in {nc_file.name}")
        print(f"    Found: {available[:10]}")
        ds.close()
        return records

    # Get time coordinates
    times = pd.to_datetime(ds['time'].values) if 'time' in ds.coords else []

    for lat, lon, name, elev in stations:
        try:
            # Nearest neighbor
            point = ds.sel(latitude=lat, longitude=lon, method='nearest')

            for i, t in enumerate(times):
                if t.hour not in TIME_STEPS:
                    continue

                # Extract pressure profiles
                temp_prof = point[temp_var].isel(time=i).values
                rh_prof = point[rh_var].isel(time=i).values
                z_prof = point[z_var].isel(time=i).values

                temp_prof = np.asarray(temp_prof).flatten()
                rh_prof = np.asarray(rh_prof).flatten()
                z_prof = np.asarray(z_prof).flatten()
                h_prof = geopotential_to_height(z_prof, lat)

                tm = compute_tm_from_profile(temp_prof, rh_prof, h_prof)

                # Surface variables
                ts = point[ts_var].isel(time=i).values if ts_var else np.nan
                ps = point[sp_var].isel(time=i).values if sp_var else np.nan
                td = point[td_var].isel(time=i).values if td_var else np.nan

                if not np.isnan(ts) and not np.isnan(td):
                    rhs = compute_rh_from_dewpoint(float(ts), float(td))
                elif len(rh_prof) > 0:
                    rhs = float(rh_prof[0])
                else:
                    rhs = np.nan

                ts_k = float(ts) if not np.isnan(ts) else np.nan
                ps_hpa = float(ps) / 100.0 if not np.isnan(ps) else np.nan

                if not np.isnan(tm) and not np.isnan(ts_k):
                    records.append({
                        'time': t, 'lat': lat, 'lon': lon, 'point_name': name,
                        'tm_k': round(float(tm), 3), 'ts_k': round(ts_k, 3),
                        'ps_hpa': round(ps_hpa, 2) if not np.isnan(ps_hpa) else np.nan,
                        'rhs_pct': round(rhs, 2) if not np.isnan(rhs) else np.nan,
                        'elevation_m': elev, 'year': t.year, 'month': t.month,
                        'day': t.day, 'hour': t.hour, 'doy': t.dayofyear,
                    })
        except Exception as e:
            continue

    ds.close()
    return records


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("NIGERIA GNSS METEOROLOGY — NCAR GDEX ERA5 DOWNLOAD")
    print("Source: NSF NCAR GDEX (Real ERA5, No Queue)")
    print("Period: 2017-2022 | Stations: 37 | Resolution: 6-hourly")
    print("=" * 70)

    print("\n⚠️  NOTE: This downloads ~2GB of NetCDF files per year.")
    print("   Total: ~12GB for 2017-2022. Ensure you have disk space.\n")

    all_records = []
    total_days = 0
    downloaded_days = 0

    for year in YEARS:
        print(f"\n[Year {year}] Downloading daily files...")
        year_records = []

        for month in range(1, 13):
            days_in_month = pd.Period(f'{year}-{month:02d}').days_in_month
            for day in range(1, days_in_month + 1):
                total_days += 1

                # Download file
                nc_file = download_ncar_file(year, month, day, RAW_DIR / str(year))
                if nc_file:
                    downloaded_days += 1
                    # Extract data
                    recs = extract_from_ncar_file(nc_file, NIGERIA_STATIONS)
                    year_records.extend(recs)

                    # Optional: Delete raw file to save space after extraction
                    # nc_file.unlink()

        all_records.extend(year_records)
        print(f"  Year {year}: {len(year_records)} records extracted")

    if not all_records:
        print("\n[ERROR] No records extracted. NCAR GDEX may be unavailable.")
        print("        Try: python 1_eras_processor_batch.py (CDS fallback)")
        return

    df = pd.DataFrame(all_records)
    df = df.sort_values(['time', 'point_name']).reset_index(drop=True)

    output_file = PROCESSED_DIR / 'nigeria_tm_training_2017_2022.csv'
    df.to_csv(output_file, index=False)

    meta = {
        "source": "NCAR_GDEX_ERA5",
        "url": "https://gdex.ucar.edu/datasets/d633000/",
        "generated": datetime.now().isoformat(),
        "records": len(df),
        "stations": int(df['point_name'].nunique()),
        "years": sorted(df['year'].unique().tolist()),
        "time_range": f"{df['time'].min()} to {df['time'].max()}",
        "tm_range": [round(df['tm_k'].min(), 2), round(df['tm_k'].max(), 2)],
        "download_success_rate": f"{downloaded_days}/{total_days} days",
    }
    meta_file = PROCESSED_DIR / 'data_source_meta.json'
    with open(meta_file, 'w') as f:
        json.dump(meta, f, indent=2)

    print("\n" + "=" * 70)
    print("✓ REAL ERA5 DATA EXTRACTION COMPLETE (NCAR GDEX)")
    print("=" * 70)
    print(f"Total records:    {len(df):,}")
    print(f"Stations:         {df['point_name'].nunique()}")
    print(f"Time range:       {df['time'].min()} to {df['time'].max()}")
    print(f"Years:            {df['year'].min()}–{df['year'].max()}")
    print(f"Tm range:         {df['tm_k'].min():.2f} K – {df['tm_k'].max():.2f} K")
    print(f"Downloaded:       {downloaded_days}/{total_days} days")
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