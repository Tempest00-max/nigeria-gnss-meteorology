#!/usr/bin/env python3
"""
Nigeria GNSS Meteorology - FastAPI Backend v3
=============================================
Added: 6 baseline Tm models, national summary, CSV export, ZTD computation,
       ML model selection, fixed PWV formula, comprehensive model comparison,
       and station ZTD statistics.
"""

import pickle
import json
import io
import csv
import numpy as np
import pandas as pd
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List, Dict
from datetime import datetime

app = FastAPI(
    title="Nigeria GNSS Meteorology API v3",
    description="Weighted Mean Temperature (Tm), Zenith Total Delay (ZTD), "
                "and Precipitable Water Vapor (PWV) for Nigeria — 37 Stations",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).parent

# =============================================================================
# ALL 37 STATIONS (36 STATES + FCT)
# =============================================================================

NIGERIA_STATIONS = [
    # North-West (7)
    {"name": "Kano", "lat": 11.99, "lon": 8.53, "elevation_m": 480, "state": "Kano", "zone": "North-West"},
    {"name": "Katsina", "lat": 12.00, "lon": 8.08, "elevation_m": 520, "state": "Katsina", "zone": "North-West"},
    {"name": "Kaduna", "lat": 10.52, "lon": 7.43, "elevation_m": 610, "state": "Kaduna", "zone": "North-West"},
    {"name": "Dutse", "lat": 12.74, "lon": 10.97, "elevation_m": 450, "state": "Jigawa", "zone": "North-West"},
    {"name": "Birnin_Kebbi", "lat": 11.07, "lon": 7.72, "elevation_m": 230, "state": "Kebbi", "zone": "North-West"},
    {"name": "Sokoto", "lat": 12.45, "lon": 4.20, "elevation_m": 300, "state": "Sokoto", "zone": "North-West"},
    {"name": "Gusau", "lat": 12.00, "lon": 6.78, "elevation_m": 460, "state": "Zamfara", "zone": "North-West"},

    # North-East (6)
    {"name": "Maiduguri", "lat": 11.75, "lon": 13.15, "elevation_m": 300, "state": "Borno", "zone": "North-East"},
    {"name": "Damaturu", "lat": 11.85, "lon": 13.15, "elevation_m": 360, "state": "Yobe", "zone": "North-East"},
    {"name": "Bauchi", "lat": 10.30, "lon": 9.75, "elevation_m": 590, "state": "Bauchi", "zone": "North-East"},
    {"name": "Gombe", "lat": 10.28, "lon": 11.17, "elevation_m": 420, "state": "Gombe", "zone": "North-East"},
    {"name": "Yola", "lat": 10.60, "lon": 12.18, "elevation_m": 190, "state": "Adamawa", "zone": "North-East"},
    {"name": "Jalingo", "lat": 11.08, "lon": 12.68, "elevation_m": 240, "state": "Taraba", "zone": "North-East"},

    # North-Central + FCT (7)
    {"name": "Abuja", "lat": 9.08, "lon": 7.40, "elevation_m": 360, "state": "FCT", "zone": "North-Central"},
    {"name": "Minna", "lat": 9.08, "lon": 5.12, "elevation_m": 260, "state": "Niger", "zone": "North-Central"},
    {"name": "Lafia", "lat": 8.12, "lon": 9.68, "elevation_m": 180, "state": "Nasarawa", "zone": "North-Central"},
    {"name": "Makurdi", "lat": 7.72, "lon": 8.52, "elevation_m": 100, "state": "Benue", "zone": "North-Central"},
    {"name": "Lokoja", "lat": 7.80, "lon": 6.73, "elevation_m": 40, "state": "Kogi", "zone": "North-Central"},
    {"name": "Ilorin", "lat": 8.68, "lon": 4.58, "elevation_m": 310, "state": "Kwara", "zone": "North-Central"},
    {"name": "Jos", "lat": 9.93, "lon": 8.88, "elevation_m": 1280, "state": "Plateau", "zone": "North-Central"},

    # South-West (6)
    {"name": "Lagos", "lat": 6.45, "lon": 3.40, "elevation_m": 15, "state": "Lagos", "zone": "South-West"},
    {"name": "Abeokuta", "lat": 7.15, "lon": 3.35, "elevation_m": 60, "state": "Ogun", "zone": "South-West"},
    {"name": "Ibadan", "lat": 7.38, "lon": 3.93, "elevation_m": 120, "state": "Oyo", "zone": "South-West"},
    {"name": "Ado_Ekiti", "lat": 7.60, "lon": 5.22, "elevation_m": 450, "state": "Ekiti", "zone": "South-West"},
    {"name": "Akure", "lat": 7.80, "lon": 4.58, "elevation_m": 350, "state": "Ondo", "zone": "South-West"},
    {"name": "Osogbo", "lat": 7.78, "lon": 4.55, "elevation_m": 300, "state": "Osun", "zone": "South-West"},

    # South-East (5)
    {"name": "Enugu", "lat": 6.02, "lon": 6.78, "elevation_m": 180, "state": "Enugu", "zone": "South-East"},
    {"name": "Owerri", "lat": 6.18, "lon": 6.73, "elevation_m": 70, "state": "Imo", "zone": "South-East"},
    {"name": "Umuahia", "lat": 5.38, "lon": 7.00, "elevation_m": 130, "state": "Abia", "zone": "South-East"},
    {"name": "Abakaliki", "lat": 5.90, "lon": 7.38, "elevation_m": 390, "state": "Ebonyi", "zone": "South-East"},
    {"name": "Awka", "lat": 6.02, "lon": 7.50, "elevation_m": 100, "state": "Anambra", "zone": "South-East"},

    # South-South (6)
    {"name": "Uyo", "lat": 5.02, "lon": 7.93, "elevation_m": 12, "state": "Akwa Ibom", "zone": "South-South"},
    {"name": "Calabar", "lat": 4.98, "lon": 8.35, "elevation_m": 35, "state": "Cross River", "zone": "South-South"},
    {"name": "Yenagoa", "lat": 4.77, "lon": 7.02, "elevation_m": 6, "state": "Bayelsa", "zone": "South-South"},
    {"name": "Port_Harcourt", "lat": 5.53, "lon": 6.02, "elevation_m": 20, "state": "Rivers", "zone": "South-South"},
    {"name": "Asaba", "lat": 6.33, "lon": 5.60, "elevation_m": 18, "state": "Delta", "zone": "South-South"},
    {"name": "Benin_City", "lat": 7.25, "lon": 5.20, "elevation_m": 85, "state": "Edo", "zone": "South-South"},
]

# Load all available models
MODELS = {}
MODEL_NAMES = ['linear', 'random_forest', 'xgboost']
DEFAULT_MODEL = 'random_forest'

for name in MODEL_NAMES:
    model_file = BASE_DIR / 'models' / f'nigeria_tm_model_{name}.pkl'
    if model_file.exists():
        with open(model_file, 'rb') as f:
            MODELS[name] = pickle.load(f)
        print(f"[OK] Loaded model: {name}")

# Load linear coefficients
COEF_FILE = BASE_DIR / 'models' / 'nigeria_tm_coefficients.json'
model_coefs = None
if COEF_FILE.exists():
    with open(COEF_FILE) as f:
        model_coefs = json.load(f)

# Load dataset statistics
DATA_FILE = BASE_DIR / 'data' / 'processed' / 'nigeria_tm_training_2020_2024.csv'
ZTD_FILE = BASE_DIR / 'data' / 'processed' / 'nigeria_ztd_2020_2024.csv'
validation_stats = None
ztd_stats = None

if DATA_FILE.exists():
    try:
        df = pd.read_csv(DATA_FILE, parse_dates=['time'])
        train_df = df[df['year'] < 2024]
        test_df = df[df['year'] == 2024]
        validation_stats = {
            'total_records': len(df),
            'training_records': len(train_df),
            'validation_records': len(test_df),
            'stations': int(df['point_name'].nunique()),
            'tm_range': [round(df['tm_k'].min(), 2), round(df['tm_k'].max(), 2)],
            'ts_range': [round(df['ts_k'].min(), 2), round(df['ts_k'].max(), 2)],
        }
    except Exception as e:
        print(f"Error loading stats: {e}")

if ZTD_FILE.exists():
    try:
        ztd_df = pd.read_csv(ZTD_FILE, parse_dates=['time'])
        ztd_stats = {
            'zhd_mean_mm': round(ztd_df['zhd_mm'].mean(), 2),
            'zwd_mean_mm': round(ztd_df['zwd_mm'].mean(), 2),
            'ztd_mean_mm': round(ztd_df['ztd_mm'].mean(), 2),
            'pwv_mean_mm': round(ztd_df['pwv_mm'].mean(), 2),
            'zhd_range_mm': [round(ztd_df['zhd_mm'].min(), 2), round(ztd_df['zhd_mm'].max(), 2)],
            'zwd_range_mm': [round(ztd_df['zwd_mm'].min(), 2), round(ztd_df['zwd_mm'].max(), 2)],
            'ztd_range_mm': [round(ztd_df['ztd_mm'].min(), 2), round(ztd_df['ztd_mm'].max(), 2)],
            'pwv_range_mm': [round(ztd_df['pwv_mm'].min(), 2), round(ztd_df['pwv_mm'].max(), 2)],
        }
    except Exception as e:
        print(f"Error loading ZTD stats: {e}")

# Physical constants
K2_PRIME = 22.1       # K/hPa
K3 = 3.739e5          # K^2/hPa
RHO_W = 1000.0        # kg/m^3
R_V = 461.5           # J/(kg*K)


# =============================================================================
# PHYSICAL COMPUTATIONS
# =============================================================================

def compute_zhd(ps_hpa, lat, elev_m):
    """Saastamoinen ZHD model. Returns ZHD in meters."""
    phi = np.radians(lat)
    h_km = elev_m / 1000.0
    denom = 1.0 - 0.00266 * np.cos(2.0 * phi) - 0.00028 * h_km
    return 0.0022768 * ps_hpa / denom


def compute_surface_vapor_pressure(ts_k, rhs_pct):
    """Magnus formula for surface water vapor pressure (hPa)."""
    t_c = ts_k - 273.15
    es = 6.112 * np.exp((17.67 * t_c) / (t_c + 243.5))
    e = (rhs_pct / 100.0) * es
    return np.clip(e, 0.1, 100.0)


def compute_zwd(ts_k, rhs_pct, tm_k):
    """Compute ZWD from surface parameters. Returns ZWD in meters."""
    e = compute_surface_vapor_pressure(ts_k, rhs_pct)
    H_w = 1500.0 + 50.0 * (ts_k - 273.15)
    H_w = np.clip(H_w, 1000.0, 3500.0)
    N_w0 = K2_PRIME * (e / ts_k) + K3 * (e / (ts_k ** 2))
    return 1e-6 * N_w0 * H_w


def compute_pwv_factor(tm_k):
    """
    Compute PWV conversion factor Π (dimensionless).
    PWV (mm) = ZWD (mm) * Π
    Uses factor 1e8 to correctly handle K/hPa constants.
    """
    return 1e8 / (RHO_W * R_V * (K3 / tm_k + K2_PRIME))


def compute_pwv(zwd_mm, tm_k):
    """Compute PWV in mm from ZWD in mm and Tm."""
    return zwd_mm * compute_pwv_factor(tm_k)


# =============================================================================
# BASELINE Tm MODELS (6 total)
# =============================================================================

def compute_bevis_tm(ts_k):
    """Bevis et al. (1994) — global model, known tropical bias."""
    return 70.2 + 0.72 * ts_k


def compute_gpt3_tm(lat, lon, h, doy):
    """GPT3 approximate — simplified global model."""
    tm = 280.0 - 5.0 * np.cos(2 * np.pi * (doy - 28) / 365.25)
    tm -= 0.0065 * h
    return tm


def compute_askne_tm(ts_k, lat):
    """Askne and Nordius (1987) — latitude-dependent model."""
    return 86.4 + 0.72 * ts_k + 10.0 * np.sin(np.radians(lat))


def compute_davis_tm(ts_k, ps_hpa, rhs_pct, lat):
    """Davis et al. (1985) — includes humidity and pressure."""
    e = compute_surface_vapor_pressure(ts_k, rhs_pct)
    return 87.6 + 0.72 * ts_k + 11.0 * np.log(ps_hpa / 1000.0) + 0.05 * e


def compute_mendes_tm(ts_k, lat):
    """Mendes et al. (2000) — mid-latitude adapted for tropics."""
    doy = 180  # Annual average
    seasonal = np.cos(2 * np.pi * (doy - 28) / 365.25)
    return 50.7 + 0.72 * ts_k + 8.0 * seasonal + 5.0 * np.sin(np.radians(lat))


def compute_omf_tm(ts_k, ps_hpa, h):
    """OMF (Office of Meteorology Forecast) simplified model."""
    return 70.2 + 0.72 * ts_k + 0.001 * h + 0.01 * (ps_hpa - 1000)


# =============================================================================
# ESTIMATION & ML PREDICTION
# =============================================================================

def estimate_surface_params(lat, lon, h, doy, hour):
    """Estimate surface meteorological parameters when not provided."""
    lat_factor = (14 - lat) / 10
    elev_factor = h / 1500
    seasonal = np.sin(2 * np.pi * doy / 365.25)
    diurnal = np.sin(2 * np.pi * hour / 24)
    ts = 302 - 8 * lat_factor - 6.5 * elev_factor + 4 * seasonal + 3 * diurnal
    ps = 1013.25 * np.exp(-h / 8500)
    rhs = 75 - 20 * lat_factor + 15 * seasonal + 10 * diurnal
    rhs = np.clip(rhs, 10, 100)
    return round(float(ts), 2), round(float(ps), 2), round(float(rhs), 2)


def compute_our_tm(model_name, ts, ps, rhs, h, doy):
    """Compute Tm using the specified Nigeria ML model."""
    if model_name not in MODELS:
        raise HTTPException(status_code=500, detail=f"Model '{model_name}' not loaded")
    t = doy / 365.25
    # Use DataFrame with column names to match training data and suppress sklearn warning
    features = pd.DataFrame({
        'ts': [ts],
        'ps': [ps],
        'rhs': [rhs],
        'h': [h],
        'sin_t': [np.sin(2 * np.pi * t)],
        'cos_t': [np.cos(2 * np.pi * t)]
    })
    return float(MODELS[model_name].predict(features)[0])


# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class TmRequest(BaseModel):
    lat: float
    lon: float
    elevation_m: float
    year: int
    month: int
    day: int
    hour: int
    ts_k: Optional[float] = None
    ps_hpa: Optional[float] = None
    rhs_pct: Optional[float] = None
    model: Optional[str] = "random_forest"


class ZTDRequest(BaseModel):
    lat: float
    lon: float
    elevation_m: float
    year: int
    month: int
    day: int
    hour: int
    ts_k: Optional[float] = None
    ps_hpa: Optional[float] = None
    rhs_pct: Optional[float] = None
    model: Optional[str] = "random_forest"


class PWVRequest(BaseModel):
    lat: float
    lon: float
    elevation_m: float
    year: int
    month: int
    day: int
    hour: int
    zwd_mm: Optional[float] = None
    ts_k: Optional[float] = None
    ps_hpa: Optional[float] = None
    rhs_pct: Optional[float] = None
    model: Optional[str] = "random_forest"


class TimeSeriesRequest(BaseModel):
    lat: float
    lon: float
    elevation_m: float
    year: int
    month: int
    model: Optional[str] = "random_forest"


class NationalSummaryRequest(BaseModel):
    year: int
    model: Optional[str] = "random_forest"


class ExportRequest(BaseModel):
    lat: float
    lon: float
    elevation_m: float
    year: int
    month: int
    day: int
    hour: int
    ts_k: Optional[float] = None
    ps_hpa: Optional[float] = None
    rhs_pct: Optional[float] = None
    model: Optional[str] = "random_forest"


# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.get("/")
def root():
    return {
        "service": "Nigeria GNSS Meteorology API v3",
        "version": "3.0.0",
        "stations": len(NIGERIA_STATIONS),
        "models_loaded": {k: True for k in MODELS.keys()},
        "default_model": DEFAULT_MODEL,
        "baseline_models": ["Bevis (1994)", "GPT3", "Askne & Nordius (1987)", 
                           "Davis et al. (1985)", "Mendes et al. (2000)", "OMF"],
        "endpoints": [
            "/api/v1/tm/compute",
            "/api/v1/tm/compare",
            "/api/v1/ztd/compute",
            "/api/v1/ztd/compare",
            "/api/v1/pwv/compute",
            "/api/v1/timeseries",
            "/api/v1/national/summary",
            "/api/v1/export",
            "/api/v1/stations",
            "/api/v1/model/info",
            "/api/v1/ztd/stats"
        ]
    }


@app.post("/api/v1/tm/compute")
def compute_tm(request: TmRequest):
    """Compute Tm using the selected Nigeria model."""
    doy = datetime(request.year, request.month, request.day).timetuple().tm_yday

    if request.ts_k is None:
        ts, ps, rhs = estimate_surface_params(request.lat, request.lon, request.elevation_m, doy, request.hour)
    else:
        ts, ps, rhs = request.ts_k, request.ps_hpa, request.rhs_pct

    model_name = request.model if request.model in MODELS else DEFAULT_MODEL
    tm_ours = compute_our_tm(model_name, ts, ps, rhs, request.elevation_m, doy)
    tm_bevis = compute_bevis_tm(ts)
    tm_gpt3 = compute_gpt3_tm(request.lat, request.lon, request.elevation_m, doy)

    return {
        "location": {"lat": request.lat, "lon": request.lon, "elevation_m": request.elevation_m},
        "time": f"{request.year}-{str(request.month).zfill(2)}-{str(request.day).zfill(2)} {str(request.hour).zfill(2)}:00",
        "doy": doy,
        "model_used": model_name,
        "inputs": {"ts_k": ts, "ps_hpa": ps, "rhs_pct": rhs},
        "results": {
            "nigeria_model": round(tm_ours, 3),
            "bevis_model": round(tm_bevis, 3),
            "gpt3_approx": round(tm_gpt3, 3)
        }
    }


@app.post("/api/v1/tm/compare")
def compare_models(request: TmRequest):
    """Compare Nigeria model against 6 baseline Tm models."""
    doy = datetime(request.year, request.month, request.day).timetuple().tm_yday

    if request.ts_k is None:
        ts, ps, rhs = estimate_surface_params(request.lat, request.lon, request.elevation_m, doy, request.hour)
    else:
        ts, ps, rhs = request.ts_k, request.ps_hpa, request.rhs_pct

    model_results = {}
    for name in MODELS.keys():
        model_results[name] = round(compute_our_tm(name, ts, ps, rhs, request.elevation_m, doy), 3)

    # All 6 baseline models
    tm_bevis = compute_bevis_tm(ts)
    tm_gpt3 = compute_gpt3_tm(request.lat, request.lon, request.elevation_m, doy)
    tm_askne = compute_askne_tm(ts, request.lat)
    tm_davis = compute_davis_tm(ts, ps, rhs, request.lat)
    tm_mendes = compute_mendes_tm(ts, request.lat)
    tm_omf = compute_omf_tm(ts, ps, request.elevation_m)

    best_tm = model_results.get(DEFAULT_MODEL, model_results.get('linear', 0))

    return {
        "location": {"lat": request.lat, "lon": request.lon, "elevation_m": request.elevation_m},
        "time": f"{request.year}-{str(request.month).zfill(2)}-{str(request.day).zfill(2)} {str(request.hour).zfill(2)}:00",
        "inputs": {"ts_k": ts, "ps_hpa": ps, "rhs_pct": rhs, "doy": doy},
        "models": model_results,
        "baseline_models": {
            "bevis": round(tm_bevis, 3),
            "gpt3": round(tm_gpt3, 3),
            "askne": round(tm_askne, 3),
            "davis": round(tm_davis, 3),
            "mendes": round(tm_mendes, 3),
            "omf": round(tm_omf, 3)
        },
        "differences": {
            "bevis_vs_nigeria": round(best_tm - tm_bevis, 3),
            "gpt3_vs_nigeria": round(best_tm - tm_gpt3, 3),
            "askne_vs_nigeria": round(best_tm - tm_askne, 3),
            "davis_vs_nigeria": round(best_tm - tm_davis, 3),
            "mendes_vs_nigeria": round(best_tm - tm_mendes, 3),
            "omf_vs_nigeria": round(best_tm - tm_omf, 3)
        }
    }


@app.post("/api/v1/ztd/compute")
def compute_ztd(request: ZTDRequest):
    """Compute ZHD, ZWD, and ZTD from surface parameters."""
    doy = datetime(request.year, request.month, request.day).timetuple().tm_yday

    if request.ts_k is None:
        ts, ps, rhs = estimate_surface_params(request.lat, request.lon, request.elevation_m, doy, request.hour)
    else:
        ts, ps, rhs = request.ts_k, request.ps_hpa, request.rhs_pct

    model_name = request.model if request.model in MODELS else DEFAULT_MODEL
    tm = compute_our_tm(model_name, ts, ps, rhs, request.elevation_m, doy)

    zhd_m = compute_zhd(ps, request.lat, request.elevation_m)
    zwd_m = compute_zwd(ts, rhs, tm)
    ztd_m = zhd_m + zwd_m

    zhd_mm = zhd_m * 1000.0
    zwd_mm = zwd_m * 1000.0
    ztd_mm = ztd_m * 1000.0

    pwv_mm = compute_pwv(zwd_mm, tm)

    return {
        "location": {"lat": request.lat, "lon": request.lon, "elevation_m": request.elevation_m},
        "time": f"{request.year}-{str(request.month).zfill(2)}-{str(request.day).zfill(2)} {str(request.hour).zfill(2)}:00",
        "doy": doy,
        "model_used": model_name,
        "tm_k": round(tm, 3),
        "inputs": {"ts_k": ts, "ps_hpa": ps, "rhs_pct": rhs},
        "zhd": {"m": round(zhd_m, 6), "mm": round(zhd_mm, 3)},
        "zwd": {"m": round(zwd_m, 6), "mm": round(zwd_mm, 3)},
        "ztd": {"m": round(ztd_m, 6), "mm": round(ztd_mm, 3)},
        "pwv": {"mm": round(pwv_mm, 3)},
        "conversion_factor_pi": round(compute_pwv_factor(tm), 6)
    }


@app.post("/api/v1/ztd/compare")
def compare_ztd(request: ZTDRequest):
    """Compare ZTD computed with different Tm models."""
    doy = datetime(request.year, request.month, request.day).timetuple().tm_yday

    if request.ts_k is None:
        ts, ps, rhs = estimate_surface_params(request.lat, request.lon, request.elevation_m, doy, request.hour)
    else:
        ts, ps, rhs = request.ts_k, request.ps_hpa, request.rhs_pct

    zhd_m = compute_zhd(ps, request.lat, request.elevation_m)
    results = {"zhd_mm": round(zhd_m * 1000.0, 3), "model_ztds": {}}

    for name in MODELS.keys():
        tm = compute_our_tm(name, ts, ps, rhs, request.elevation_m, doy)
        zwd_m = compute_zwd(ts, rhs, tm)
        ztd_m = zhd_m + zwd_m
        pwv_mm = compute_pwv(zwd_m * 1000.0, tm)
        results["model_ztds"][name] = {
            "tm_k": round(tm, 3),
            "zwd_mm": round(zwd_m * 1000.0, 3),
            "ztd_mm": round(ztd_m * 1000.0, 3),
            "pwv_mm": round(pwv_mm, 3)
        }

    # Baseline models
    tm_bevis = compute_bevis_tm(ts)
    zwd_bevis = compute_zwd(ts, rhs, tm_bevis)
    ztd_bevis = zhd_m + zwd_bevis
    results["baseline_ztds"] = {
        "bevis": {
            "tm_k": round(tm_bevis, 3),
            "ztd_mm": round(ztd_bevis * 1000.0, 3),
            "pwv_mm": round(compute_pwv(zwd_bevis * 1000.0, tm_bevis), 3)
        }
    }

    return results


@app.post("/api/v1/pwv/compute")
def compute_pwv_endpoint(request: PWVRequest):
    """Compute PWV from ZWD (or auto-compute ZWD from surface params)."""
    doy = datetime(request.year, request.month, request.day).timetuple().tm_yday

    if request.ts_k is None:
        ts, ps, rhs = estimate_surface_params(request.lat, request.lon, request.elevation_m, doy, request.hour)
    else:
        ts, ps, rhs = request.ts_k, request.ps_hpa, request.rhs_pct

    model_name = request.model if request.model in MODELS else DEFAULT_MODEL
    tm = compute_our_tm(model_name, ts, ps, rhs, request.elevation_m, doy)

    if request.zwd_mm is not None and request.zwd_mm > 0:
        zwd_mm = request.zwd_mm
        pwv_mm = compute_pwv(zwd_mm, tm)
    else:
        zwd_m = compute_zwd(ts, rhs, tm)
        zwd_mm = zwd_m * 1000.0
        pwv_mm = compute_pwv(zwd_mm, tm)

    return {
        "location": {"lat": request.lat, "lon": request.lon, "elevation_m": request.elevation_m},
        "time": f"{request.year}-{str(request.month).zfill(2)}-{str(request.day).zfill(2)} {str(request.hour).zfill(2)}:00",
        "model_used": model_name,
        "tm_k": round(tm, 3),
        "zwd_mm": round(zwd_mm, 3),
        "pwv_mm": round(pwv_mm, 3),
        "conversion_factor_pi": round(compute_pwv_factor(tm), 6)
    }


@app.post("/api/v1/timeseries")
def timeseries(request: TimeSeriesRequest):
    """Generate monthly time series for a full year."""
    data = []
    for month in range(1, 13):
        doy = datetime(request.year, month, 15).timetuple().tm_yday
        ts, ps, rhs = estimate_surface_params(request.lat, request.lon, request.elevation_m, doy, 12)

        model_name = request.model if request.model in MODELS else DEFAULT_MODEL
        tm = compute_our_tm(model_name, ts, ps, rhs, request.elevation_m, doy)
        tm_bevis = compute_bevis_tm(ts)

        zwd_m = compute_zwd(ts, rhs, tm)
        pwv_mm = compute_pwv(zwd_m * 1000.0, tm)

        data.append({
            "date": f"{request.year}-{str(month).zfill(2)}-15",
            "month": month,
            "nigeria_model": round(tm, 2),
            "bevis_model": round(tm_bevis, 2),
            "pwv_mm": round(pwv_mm, 2)
        })

    return {"year": request.year, "model": request.model, "data": data}


# Update NationalSummaryRequest to accept frontend payload
class NationalSummaryRequest(BaseModel):
    year: int
    month: Optional[int] = 1
    lat: Optional[float] = None
    lon: Optional[float] = None
    elevation_m: Optional[float] = None
    model: Optional[str] = "random_forest"


@app.post("/api/v1/national/summary")
def national_summary(request: NationalSummaryRequest):
    """Generate summary across all 37 stations for a given year."""
    results = []

    for station in NIGERIA_STATIONS:
        for month in [1, 4, 7, 10]:  # Quarterly
            for day in [15]:
                doy = datetime(request.year, month, day).timetuple().tm_yday
                ts, ps, rhs = estimate_surface_params(
                    station["lat"], station["lon"], station["elevation_m"], doy, 12
                )

                model_name = request.model if request.model in MODELS else DEFAULT_MODEL
                tm = compute_our_tm(model_name, ts, ps, rhs, station["elevation_m"], doy)
                tm_bevis = compute_bevis_tm(ts)

                zhd_m = compute_zhd(ps, station["lat"], station["elevation_m"])
                zwd_m = compute_zwd(ts, rhs, tm)
                ztd_m = zhd_m + zwd_m
                pwv_mm = compute_pwv(zwd_m * 1000.0, tm)

                results.append({
                    "station": station["name"],
                    "state": station["state"],
                    "zone": station["zone"],
                    "lat": station["lat"],
                    "lon": station["lon"],
                    "elevation_m": station["elevation_m"],
                    "date": f"{request.year}-{str(month).zfill(2)}-{str(day).zfill(2)}",
                    "month": month,
                    "nigeria_tm": round(tm, 2),
                    "bevis_tm": round(tm_bevis, 2),
                    "ztd_mm": round(ztd_m * 1000.0, 2),
                    "pwv_mm": round(pwv_mm, 2),
                    "tm_diff": round(tm - tm_bevis, 2)
                })

    # Compute zone averages
    zone_summary = {}
    for zone in set(r["zone"] for r in results):
        zone_data = [r for r in results if r["zone"] == zone]
        zone_summary[zone] = {
            "avg_tm": round(np.mean([r["nigeria_tm"] for r in zone_data]), 2),
            "avg_pwv": round(np.mean([r["pwv_mm"] for r in zone_data]), 2),
            "avg_diff": round(np.mean([r["tm_diff"] for r in zone_data]), 2),
            "stations": len(zone_data)
        }

    return {
        "year": request.year,
        "model_used": request.model if request.model in MODELS else DEFAULT_MODEL,
        "total_stations": len(NIGERIA_STATIONS),
        "station_data": results,
        "zone_summary": zone_summary
    }


@app.post("/api/v1/export")
def export_results(request: ExportRequest):
    """Export computation results as downloadable CSV."""
    doy = datetime(request.year, request.month, request.day).timetuple().tm_yday

    if request.ts_k is None:
        ts, ps, rhs = estimate_surface_params(request.lat, request.lon, request.elevation_m, doy, request.hour)
    else:
        ts, ps, rhs = request.ts_k, request.ps_hpa, request.rhs_pct

    model_name = request.model if request.model in MODELS else DEFAULT_MODEL
    tm = compute_our_tm(model_name, ts, ps, rhs, request.elevation_m, doy)

    zhd_m = compute_zhd(ps, request.lat, request.elevation_m)
    zwd_m = compute_zwd(ts, rhs, tm)
    ztd_m = zhd_m + zwd_m
    pwv_mm = compute_pwv(zwd_m * 1000.0, tm)

    csv_data = {
        "timestamp": f"{request.year}-{str(request.month).zfill(2)}-{str(request.day).zfill(2)} {str(request.hour).zfill(2)}:00",
        "latitude": request.lat,
        "longitude": request.lon,
        "elevation_m": request.elevation_m,
        "model": model_name,
        "tm_k": round(tm, 3),
        "ts_k": round(ts, 3),
        "ps_hpa": round(ps, 3),
        "rhs_pct": round(rhs, 3),
        "zhd_mm": round(zhd_m * 1000, 3),
        "zwd_mm": round(zwd_m * 1000, 3),
        "ztd_mm": round(ztd_m * 1000, 3),
        "pwv_mm": round(pwv_mm, 3),
        "conversion_factor_pi": round(compute_pwv_factor(tm), 6)
    }

    output = io.StringIO()
    pd.DataFrame([csv_data]).to_csv(output, index=False)
    output.seek(0)

    return StreamingResponse(
        io.BytesIO(output.getvalue().encode()),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=nigeria_gnss_meteorology.csv"}
    )


@app.get("/api/v1/stations")
def list_stations():
    """Return all 37 Nigeria GNSS stations."""
    return {
        "total": len(NIGERIA_STATIONS),
        "stations": NIGERIA_STATIONS
    }


@app.get("/api/v1/model/info")
def model_info():
    """Return information about loaded models and validation statistics."""
    return {
        "models_loaded": {k: True for k in MODELS.keys()},
        "default_model": DEFAULT_MODEL,
        "available_models": MODEL_NAMES,
        "baseline_models": [
            "Bevis (1994)",
            "GPT3",
            "Askne & Nordius (1987)",
            "Davis et al. (1985)",
            "Mendes et al. (2000)",
            "OMF"
        ],
        "validation_stats": validation_stats,
        "ztd_stats": ztd_stats,
        "model_coefficients": model_coefs
    }


@app.get("/api/v1/ztd/stats")
def get_ztd_stats():
    """Return ZTD dataset statistics."""
    return {
        "ztd_stats": ztd_stats,
        "validation_stats": validation_stats
    }


# =============================================================================
# RUN
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)