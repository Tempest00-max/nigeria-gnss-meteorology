#!/usr/bin/env python3
"""
0_generate_synthetic_data.py — Physics-Based Synthetic Nigeria ERA5 Data
========================================================================
Generates realistic Tm, Ts, Ps, RH data for all 37 Nigerian stations.
Uses actual climate patterns: Harmattan dry season, rainy season, elevation effects.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

# =============================================================================
# CONFIGURATION
# =============================================================================

NIGERIA_POINTS = [
    # North-West (7)
    (11.99, 8.53, "Kano"), (12.00, 8.08, "Katsina"), (10.52, 7.43, "Kaduna"),
    (12.74, 10.97, "Dutse"), (11.07, 7.72, "Birnin_Kebbi"), (12.45, 4.20, "Sokoto"),
    (12.00, 6.78, "Gusau"),
    # North-East (6)
    (11.75, 13.15, "Maiduguri"), (11.85, 13.15, "Damaturu"), (10.30, 9.75, "Bauchi"),
    (10.28, 11.17, "Gombe"), (10.60, 12.18, "Yola"), (11.08, 12.68, "Jalingo"),
    # North-Central + FCT (7)
    (9.08, 7.40, "Abuja"), (9.08, 5.12, "Minna"), (8.12, 9.68, "Lafia"),
    (7.72, 8.52, "Makurdi"), (7.80, 6.73, "Lokoja"), (8.68, 4.58, "Ilorin"),
    (9.93, 8.88, "Jos"),
    # South-West (6)
    (6.45, 3.40, "Lagos"), (7.15, 3.35, "Abeokuta"), (7.38, 3.93, "Ibadan"),
    (7.60, 5.22, "Ado_Ekiti"), (7.80, 4.58, "Akure"), (7.78, 4.55, "Osogbo"),
    # South-East (5)
    (6.02, 6.78, "Enugu"), (6.18, 6.73, "Owerri"), (5.38, 7.00, "Umuahia"),
    (5.90, 7.38, "Abakaliki"), (6.02, 7.50, "Awka"),
    # South-South (6)
    (5.02, 7.93, "Uyo"), (4.98, 8.35, "Calabar"), (4.77, 7.02, "Yenagoa"),
    (5.53, 6.02, "Port_Harcourt"), (6.33, 5.60, "Asaba"), (7.25, 5.20, "Benin_City"),
]

YEARS = [2020, 2021, 2022, 2023, 2024]
TIME_STEPS = [0, 6, 12, 18]  # UTC hours

BASE_DIR = Path(__file__).parent
PROCESSED_DIR = BASE_DIR / 'data' / 'processed'
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# PHYSICS-BASED SYNTHETIC GENERATION
# =============================================================================

def generate_station_data(lat, lon, name, year):
    """Generate realistic meteorological data for one station-year."""
    records = []
    
    # Climate zone parameters
    is_north = lat > 9.0
    is_coastal = lon < 6.0 and lat < 7.0
    is_highland = name in ["Jos"]
    elevation = max(10, int((14 - lat) * 50 + np.random.normal(0, 30)))
    if is_highland:
        elevation = 1280
    
    # Base temperatures by zone
    base_ts = 305.0 if is_coastal else 300.0 if not is_north else 295.0
    if is_highland:
        base_ts = 288.0
    
    for month in range(1, 13):
        days_in_month = pd.Period(f'{year}-{month:02d}').days_in_month
        
        for day in range(1, days_in_month + 1):
            doy = datetime(year, month, day).timetuple().tm_yday
            
            # Seasonal cycle (rainy vs dry)
            # Peak rainy: July-August (doy 180-240)
            # Peak dry: December-February (doy 335-60)
            seasonal = np.sin(2 * np.pi * (doy - 120) / 365.25)
            
            # Harmattan effect (dry, dusty, high diurnal range)
            harmattan = 1.0 if month in [12, 1, 2] else 0.0
            
            for hour in TIME_STEPS:
                # Diurnal cycle
                diurnal = np.sin(2 * np.pi * (hour - 6) / 24)
                
                # Surface temperature (K)
                ts = base_ts + 4 * seasonal - 2 * harmattan + 3 * diurnal
                ts += np.random.normal(0, 1.5)  # Weather noise
                
                # Surface pressure (hPa) — elevation effect
                ps = 1013.25 * np.exp(-elevation / 8500) + np.random.normal(0, 2)
                
                # Relative humidity (%)
                # High in rainy season south, low in Harmattan north
                base_rh = 80 if is_coastal else 70 if not is_north else 50
                rh = base_rh + 15 * seasonal - 20 * harmattan + np.random.normal(0, 5)
                rh = np.clip(rh, 15, 98)
                
                # Tm (K) — physically realistic based on Ts and climate
                # Bevis-like but with regional variation
                tm = 70.2 + 0.72 * ts + 5 * seasonal
                if is_north and harmattan > 0.5:
                    tm -= 3  # Dry Harmattan air has lower Tm
                if is_coastal:
                    tm += 2  # Maritime influence
                tm += np.random.normal(0, 2.5)
                
                # Ensure Tm is physically reasonable (260-310 K)
                tm = np.clip(tm, 260, 310)
                
                records.append({
                    'time': datetime(year, month, day, hour),
                    'lat': lat,
                    'lon': lon,
                    'point_name': name,
                    'tm_k': round(tm, 3),
                    'ts_k': round(ts, 3),
                    'ps_hpa': round(ps, 2),
                    'rhs_pct': round(rh, 2),
                    'elevation_m': elevation,
                    'year': year,
                    'month': month,
                    'day': day,
                    'hour': hour,
                    'doy': doy,
                })
    
    return records


def main():
    print("Generating Physics-Based Synthetic Nigeria ERA5 Data")
    print("=" * 60)
    print(f"Stations: {len(NIGERIA_POINTS)}")
    print(f"Years: {YEARS}")
    print("This data mimics real Nigerian climate patterns")
    print("=" * 60)
    
    all_records = []
    
    for year in YEARS:
        print(f"\nGenerating {year}...")
        for lat, lon, name in NIGERIA_POINTS:
            recs = generate_station_data(lat, lon, name, year)
            all_records.extend(recs)
            print(f"  {name}: {len(recs)} records")
    
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
    print(f"Ts: {df['ts_k'].min():.1f}K to {df['ts_k'].max():.1f}K")
    print(f"Saved: {output_file}")
    print(f"{'='*60}")
    print("\n⚠️  NOTE: This is SYNTHETIC data for demonstration only.")
    print("Replace with real ERA5 data when downloads complete.")


if __name__ == '__main__':
    main()