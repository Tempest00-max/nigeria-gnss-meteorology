# NiMet-GNSS | Nigeria GNSS Meteorology System

**Autonomous GNSS-Meteorological Analysis for Regional PWV Estimation in Nigeria**

## Features
- 37 stations across all 36 Nigerian states + FCT
- Machine Learning-based Weighted Mean Temperature (Tm) model
- Real-time ZTD, ZWD, ZTD, and PWV computation
- Model comparison: Nigeria vs Bevis vs GPT3 vs 6 baselines
- Interactive dashboard with Leaflet mapping

## Models
- Linear Regression (RMSE: 2.015 K)
- Random Forest (RMSE: 2.052 K)
- XGBoost (RMSE: 2.026 K)

## Validation
- R² = 0.9008
- 90.5% improvement over Bevis global model
- 6-hourly temporal resolution, 2017-2022 coverage

## API
- POST /api/v1/tm/compute
- POST /api/v1/ztd/compute
- POST /api/v1/tm/compare
- POST /api/v1/timeseries
- POST /api/v1/national/summary

## Run Locally
```bash
pip install -r requirements.txt
python backend_v3.py
# Open 4_frontend_v3.html in browser

Author
Gbadamosi Tolulope | Geodesy MSc | Nigeria