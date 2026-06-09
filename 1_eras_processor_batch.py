#!/usr/bin/env python3
"""
Nigeria GNSS Meteorology - ERA5 Processor (MONTHLY BATCH - 37 POINTS)
=====================================================================
Downloads monthly bounding-box chunks from CDS API and extracts point data locally.
Covers all 36 Nigerian states + FCT. ~60 API calls total.
"""

import os
import sys
import cdsapi
import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import time
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

NIGERIA_POINTS = [
    # North-West (7)
    (11.99, 8.53, "Kano"),
    (12.00, 8.08, "Katsina"),
    (10.52, 7.43, "Kaduna"),
    (12.74, 10.97, "Dutse"),          # Jigawa
    (11.07, 7.72, "Birnin_Kebbi"),    # Kebbi
    (12.45, 4.20, "Sokoto"),
    (12.00, 6.78, "Gusau"),           # Zamfara

    # North-East (6)
    (11.75, 13.15, "Maiduguri"),      # Borno
    (11.85, 13.15, "Damaturu"),       # Yobe
    (10.30, 9.75, "Bauchi"),
    (10.28, 11.17, "Gombe"),
    (10.60, 12.18, "Yola"),           # Adamawa
    (11.08, 12.68, "Jalingo"),        # Taraba

    # North-Central + FCT (7)
    (9.08, 7.40, "Abuja"),            # FCT
    (9.08, 5.12, "Minna"),            # Niger
    (8.12, 9.68, "Lafia"),            # Nasarawa
    (7.72, 8.52, "Makurdi"),          # Benue
    (7.80, 6.73, "Lokoja"),           # Kogi
    (8.68, 4.58, "Ilorin"),           # Kwara
    (9.93, 8.88, "Jos"),              # Plateau

    # South-West (6)
    (6.45, 3.40, "Lagos"),
    (7.15, 3.35, "Abeokuta"),         # Ogun
    (7.38, 3.93, "Ibadan"),           # Oyo
    (7.60, 5.22, "Ado_Ekiti"),        # Ekiti
    (7.80, 4.58, "Akure"),            # Ondo
    (7.78, 4.55, "Osogbo"),           # Osun

    # South-East (5)
    (6.02, 6.78, "Enugu"),
    (6.18, 6.73, "Owerri"),           # Imo
    (5.38, 7.00, "Umuahia"),          # Abia
    (5.90, 7.38, "Abakaliki"),        # Ebonyi
    (6.02, 7.50, "Awka"),             # Anambra

    # South-South (6)
    (5.02, 7.93, "Uyo"),              # Akwa Ibom
    (4.98, 8.35, "Calabar"),          # Cross River
    (4.77, 7.02, "Yenagoa"),          # Bayelsa
    (5.53, 6.02, "Port_Harcourt"),    # Rivers
    (6.33, 5.60, "Asaba"),            # Delta
    (7.25, 5.20, "Benin_City"),       # Edo
]

# Hardcoded generous Nigeria bounding box [N, W, S, E]
# Covers all 36 states + FCT with margin for ERA5 0.25° grid
AREA = [14.0, 2.5, 4.0, 14.5]

ALL_YEARS = [2020, 2021, 2022, 2023, 2024]
PRESSURE_LEVELS = [1000, 950, 925, 900, 875, 850, 800, 750, 700, 600, 500, 400, 300]
TIME_STEPS = ['00:00', '06:00', '12:00', '18:00']
PRESSURE_VARS = ['geopotential', 'temperature', 'relative_humidity']
SURFACE_VARS = ['2m_temperature', '2m_dewpoint_temperature', 'surface_pressure']

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'data'
RAW_DIR = DATA_DIR / 'raw_era5'
PROCESSED_DIR = DATA_DIR / 'processed'

for d in [DATA_DIR, RAW_DIR, PROCESSED_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# =============================================================================
# PHYSICAL FORMULAS
# =============================================================================

def compute_rh_from_dewpoint(t_k, td_k):
    t = t_k - 273.15
    td = td_k - 273.15
    a = 17.625
    b = 243.04
    rh = 100 * np.exp((a * td / (b + td)) - (a * t / (b + t)))
    return np.clip(rh, 0.1, 100.0)


def compute_water_vapor_pressure(rh, t_celsius):
    rh = np.asarray(rh, dtype=np.float64)
    t_c = np.asarray(t_celsius, dtype=np.float64)
    rh = np.clip(rh, 0.1, 100.0)
    es = 6.112 * np.exp((17.67 * t_c) / (t_c + 243.5))
    e = (rh / 100.0) * es
    return e


def compute_tm_from_profile(temperature_k, relative_humidity, geopotential_height_m):
    temp = np.asarray(temperature_k, dtype=np.float64)
    rh = np.asarray(relative_humidity, dtype=np.float64)
    h = np.asarray(geopotential_height_m, dtype=np.float64)

    sort_idx = np.argsort(h)
    temp = temp[sort_idx]
    rh = rh[sort_idx]
    h = h[sort_idx]

    valid = ~(np.isnan(temp) | np.isnan(rh) | np.isnan(h))
    if valid.sum() < 3:
        return np.nan

    temp = temp[valid]
    rh = rh[valid]
    h = h[valid]

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
# HELPERS
# =============================================================================

def get_coord_name(ds, candidates):
    """Find the actual coordinate name in the dataset from a list of candidates."""
    for c in candidates:
        if c in ds.coords:
            return c
    for dim in ds.dims:
        if dim.lower() in [c.lower() for c in candidates]:
            return dim
    return None


def to_scalar(val):
    arr = np.asarray(val)
    if arr.size == 1:
        return float(arr.flat[0])
    elif arr.size == 0:
        return np.nan
    return float(arr.flat[0])


# =============================================================================
# CDS API DOWNLOAD (MONTHLY BATCH)
# =============================================================================

def download_pressure_month(year, month, output_dir):
    client = cdsapi.Client()
    output_file = output_dir / f"p_{year}_{month:02d}.nc"

    if output_file.exists():
        return output_file

    try:
        client.retrieve(
            'reanalysis-era5-pressure-levels',
            {
                'product_type': 'reanalysis',
                'variable': PRESSURE_VARS,
                'pressure_level': [str(p) for p in PRESSURE_LEVELS],
                'year': str(year),
                'month': f'{month:02d}',
                'day': [f'{d:02d}' for d in range(1, 32)],
                'time': TIME_STEPS,
                'area': AREA,
                'data_format': 'netcdf',
            },
            str(output_file)
        )
        return output_file
    except Exception as e:
        print(f"    [ERR] Pressure {year}-{month:02d}: {str(e)[:120]}")
        return None


def download_surface_month(year, month, output_dir):
    client = cdsapi.Client()
    output_file = output_dir / f"s_{year}_{month:02d}.nc"

    if output_file.exists():
        return output_file

    try:
        client.retrieve(
            'reanalysis-era5-single-levels',
            {
                'product_type': 'reanalysis',
                'variable': SURFACE_VARS,
                'year': str(year),
                'month': f'{month:02d}',
                'day': [f'{d:02d}' for d in range(1, 32)],
                'time': TIME_STEPS,
                'area': AREA,
                'data_format': 'netcdf',
            },
            str(output_file)
        )
        return output_file
    except Exception as e:
        print(f"    [ERR] Surface {year}-{month:02d}: {str(e)[:120]}")
        return None


# =============================================================================
# PROCESSING (LOCAL EXTRACTION FROM MONTHLY FILES)
# =============================================================================

def process_monthly_files(p_file, s_file):
    records = []

    try:
        dsp = xr.open_dataset(p_file)
        dss = xr.open_dataset(s_file)
    except Exception as e:
        print(f"    [ERR] Cannot open files: {e}")
        return records

    # Handle expver dimension if present (keep expver=1, drop the dim)
    if 'expver' in dsp.dims:
        if dsp.sizes['expver'] > 1:
            dsp = dsp.sel(expver=1)
        else:
            dsp = dsp.squeeze('expver', drop=True)
    if 'expver' in dss.dims:
        if dss.sizes['expver'] > 1:
            dss = dss.sel(expver=1)
        else:
            dss = dss.squeeze('expver', drop=True)

    # Find coordinate names
    lat_name = get_coord_name(dsp, ['latitude', 'lat'])
    lon_name = get_coord_name(dsp, ['longitude', 'lon'])
    p_time = get_coord_name(dsp, ['valid_time', 'time'])
    s_time = get_coord_name(dss, ['valid_time', 'time'])

    if None in (lat_name, lon_name, p_time, s_time):
        print(f"    [ERR] Missing coords: lat={lat_name}, lon={lon_name}, p_time={p_time}, s_time={s_time}")
        return records

    # Variable name mapping
    temp_var = 't' if 't' in dsp.data_vars else 'temperature'
    rh_var = 'r' if 'r' in dsp.data_vars else 'relative_humidity'
    z_var = 'z' if 'z' in dsp.data_vars else 'geopotential'

    ts_var = 't2m' if 't2m' in dss.data_vars else '2m_temperature'
    td_var = 'd2m' if 'd2m' in dss.data_vars else '2m_dewpoint_temperature'
    sp_var = 'sp' if 'sp' in dss.data_vars else 'surface_pressure'

    # Process each point
    for lat, lon, name in NIGERIA_POINTS:
        try:
            # Nearest-neighbor extraction from the grid
            p_point = dsp.sel({lat_name: lat, lon_name: lon}, method='nearest')
            s_point = dss.sel({lat_name: lat, lon_name: lon}, method='nearest')

            times = pd.to_datetime(p_point[p_time].values)
            n_times = len(times)

            for t_idx in range(n_times):
                try:
                    # Extract pressure profile
                    temp_prof = p_point[temp_var].isel({p_time: t_idx}).values
                    rh_prof = p_point[rh_var].isel({p_time: t_idx}).values
                    z_prof = p_point[z_var].isel({p_time: t_idx}).values

                    temp_prof = np.asarray(temp_prof).flatten()
                    rh_prof = np.asarray(rh_prof).flatten()
                    z_prof = np.asarray(z_prof).flatten()

                    h_prof = geopotential_to_height(z_prof, lat)
                    tm = compute_tm_from_profile(temp_prof, rh_prof, h_prof)

                    # Extract surface scalars
                    ts = to_scalar(s_point[ts_var].isel({s_time: t_idx}).values)
                    td = to_scalar(s_point[td_var].isel({s_time: t_idx}).values)
                    ps_pa = to_scalar(s_point[sp_var].isel({s_time: t_idx}).values)

                    ps_hpa = ps_pa / 100.0
                    rhs = compute_rh_from_dewpoint(ts, td)

                    if not np.isnan(tm):
                        t = times[t_idx]
                        records.append({
                            'time': t,
                            'lat': lat,
                            'lon': lon,
                            'point_name': name,
                            'tm_k': round(float(tm), 3),
                            'ts_k': round(ts, 3),
                            'ps_hpa': round(ps_hpa, 2),
                            'rhs_pct': round(rhs, 2),
                            'elevation_m': round(float(h_prof[0]), 1),
                            'year': t.year,
                            'month': t.month,
                            'day': t.day,
                            'hour': t.hour,
                            'doy': t.dayofyear,
                        })
                except Exception as e:
                    if t_idx == 0:
                        print(f"    [ERR] Point {name} t={t_idx}: {str(e)[:80]}")
                    continue

        except Exception as e:
            print(f"    [ERR] Point {name}: {str(e)[:80]}")
            continue

    dsp.close()
    dss.close()
    return records


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("Nigeria GNSS Meteorology - ERA5 Processor (MONTHLY BATCH - 37 POINTS)")
    print("=" * 60)
    print(f"Points: {len(NIGERIA_POINTS)} (36 states + FCT)")
    print(f"Years: {ALL_YEARS}")
    print(f"Bounding box [N,W,S,E]: {AREA}")
    print(f"Expected API calls: {len(ALL_YEARS) * 12 * 2} (pressure + surface per month)")
    print("=" * 60)

    cdsapirc = Path.home() / '.cdsapirc'
    if not cdsapirc.exists():
        print("[ERROR] CDS API credentials not found at ~/.cdsapirc")
        sys.exit(1)

    all_records = []
    total_months = len(ALL_YEARS) * 12
    month_count = 0

    for year in ALL_YEARS:
        for month in range(1, 13):
            month_count += 1
            print(f"\n[{month_count}/{total_months}] {year}-{month:02d}")

            p_file = download_pressure_month(year, month, RAW_DIR)
            s_file = download_surface_month(year, month, RAW_DIR)

            if p_file and s_file:
                recs = process_monthly_files(p_file, s_file)
                if recs:
                    all_records.extend(recs)
                    print(f"  -> {len(recs)} records (total: {len(all_records):,})")
                else:
                    print(f"  -> No records generated")
            else:
                print(f"  -> Download failed, skipping")

            # Polite delay between months
            time.sleep(1)

    if len(all_records) > 0:
        df = pd.DataFrame(all_records)
        df = df.sort_values(['time', 'point_name']).reset_index(drop=True)

        output_file = PROCESSED_DIR / 'nigeria_tm_training_2020_2024.csv'
        df.to_csv(output_file, index=False)

        print(f"\n{'='*60}")
        print(f"COMPLETE")
        print(f"Records: {len(df):,}")
        print(f"Points: {df['point_name'].nunique()}")
        print(f"Time range: {df['time'].min()} to {df['time'].max()}")
        print(f"Tm: {df['tm_k'].min():.1f}K to {df['tm_k'].max():.1f}K")
        print(f"Saved: {output_file}")
        print(f"{'='*60}")
    else:
        print("[ERROR] No records generated.")


if __name__ == '__main__':
    main()