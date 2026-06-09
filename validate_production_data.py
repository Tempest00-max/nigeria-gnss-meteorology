#!/usr/bin/env python3
"""
validate_production_data.py — Quick Validation for Real/Synthetic Data
=======================================================================
Validates ERA5 extraction quality, checks physical ranges, compares with
Bevis model, and generates a go/no-go report for thesis submission.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
from datetime import datetime

def validate():
    BASE_DIR = Path(__file__).parent
    DATA_FILE = BASE_DIR / 'data' / 'processed' / 'nigeria_tm_training_2017_2022.csv'
    META_FILE = BASE_DIR / 'data' / 'processed' / 'data_source_meta.json'

    print("=" * 70)
    print("NIGERIA GNSS METEOROLOGY — PRODUCTION DATA VALIDATION")
    print("=" * 70)

    if not DATA_FILE.exists():
        print(f"\n✗ FATAL: Data file not found: {DATA_FILE}")
        print("   Run: python 1_era5_production_2017_2022.py")
        return False

    df = pd.read_csv(DATA_FILE, parse_dates=['time'])

    # Load metadata if available
    source = "UNKNOWN"
    if META_FILE.exists():
        with open(META_FILE) as f:
            meta = json.load(f)
            source = meta.get('source', 'UNKNOWN')

    print(f"\n[1] Data Source: {source}")
    print(f"    Records: {len(df):,}")
    print(f"    Stations: {df['point_name'].nunique()}")
    print(f"    Years: {sorted(df['year'].unique())}")
    print(f"    Time range: {df['time'].min()} to {df['time'].max()}")

    # Physical range checks
    print("\n[2] Physical Range Validation")
    checks = {
        'tm_k': (200, 320, "Weighted mean temperature"),
        'ts_k': (270, 330, "Surface temperature"),
        'ps_hpa': (800, 1050, "Surface pressure"),
        'rhs_pct': (5, 100, "Relative humidity"),
    }

    all_pass = True
    for col, (min_val, max_val, name) in checks.items():
        if col in df.columns:
            actual_min = df[col].min()
            actual_max = df[col].max()
            pass_check = min_val <= actual_min and actual_max <= max_val
            status = "✓" if pass_check else "✗"
            print(f"    {status} {name}: {actual_min:.1f} – {actual_max:.1f} (expected {min_val}–{max_val})")
            if not pass_check:
                all_pass = False

    # Completeness check
    print("\n[3] Completeness Check")
    expected_per_station = len(df) // df['point_name'].nunique()
    station_counts = df.groupby('point_name').size()
    min_count = station_counts.min()
    max_count = station_counts.max()

    print(f"    Records per station: {min_count} – {max_count}")
    if min_count < expected_per_station * 0.8:
        print(f"    ⚠ Some stations have < 80% expected records")
        all_pass = False
    else:
        print(f"    ✓ All stations have sufficient records")

    # Tm vs Ts relationship (Bevis check)
    print("\n[4] Tm vs Ts Relationship")
    bevis_tm = 70.2 + 0.72 * df['ts_k']
    bevis_rmse = np.sqrt(((bevis_tm - df['tm_k']) ** 2).mean())
    bevis_bias = (bevis_tm - df['tm_k']).mean()

    print(f"    Bevis RMSE: {bevis_rmse:.2f} K")
    print(f"    Bevis Bias: {bevis_bias:.2f} K")

    if bevis_rmse > 25:
        print(f"    ⚠ Bevis RMSE suspiciously high — check Tm computation")
    else:
        print(f"    ✓ Tm values are physically consistent with Ts")

    # Station coverage
    print("\n[5] Station Coverage")
    for name, count in station_counts.sort_values(ascending=False).items():
        pct = count / len(df) * 100
        print(f"    {name:15s}: {count:5d} records ({pct:.1f}%)")

    # Seasonal check
    print("\n[6] Seasonal Tm Variation")
    monthly = df.groupby('month')['tm_k'].mean()
    for month, val in monthly.items():
        print(f"    Month {month:2d}: {val:.2f} K")

    seasonal_range = monthly.max() - monthly.min()
    print(f"    Seasonal range: {seasonal_range:.2f} K")

    if seasonal_range < 2:
        print(f"    ⚠ Seasonal variation very low — possible data issue")
    else:
        print(f"    ✓ Reasonable seasonal variation detected")

    # Final verdict
    print("\n" + "=" * 70)
    if all_pass:
        print("✓ VALIDATION PASSED — Data is ready for model training")
        print("=" * 70)
        print("\nNext steps:")
        print("   python 2_model_fitter_ML.py")
        print("   python 3_ztd_processor.py")
        print("   python report_generator_v2.py")
        return True
    else:
        print("⚠ VALIDATION WARNINGS — Review issues above before training")
        print("=" * 70)
        return False

if __name__ == '__main__':
    validate()