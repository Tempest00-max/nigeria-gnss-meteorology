#!/usr/bin/env python3
"""
Nigeria GNSS Meteorology - ML-Enhanced Model Fitter
=====================================================
Fits and compares three models:
  1. Multivariate Linear Regression (interpretable baseline)
  2. Random Forest Regressor (ensemble ML)
  3. XGBoost Regressor (gradient boosting ML)

Satisfies the "Machine Learning-Based" claim in the thesis title
while preserving the physically interpretable linear model.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import json
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("[INFO] XGBoost not installed. Install with: pip install xgboost")
    print("       Using Linear Regression + Random Forest only.")


def load_data():
    """Load processed training data."""
    BASE_DIR = Path(__file__).parent
    data_file = BASE_DIR / 'data' / 'processed' / 'nigeria_tm_training_2020_2024.csv'
    if not data_file.exists():
        print(f"[ERROR] Data file not found: {data_file}")
        print("Run 0_generate_synthetic_data.py or 1_era5_processor first.")
        return None
    df = pd.read_csv(data_file, parse_dates=['time'])
    print(f"Loaded {len(df):,} records")
    return df


def prepare_features(df):
    """Create feature matrix X and target vector y."""
    t = df['doy'] / 365.25
    X = pd.DataFrame({
        'ts': df['ts_k'],
        'ps': df['ps_hpa'],
        'rhs': df['rhs_pct'],
        'h': df['elevation_m'],
        'sin_t': np.sin(2 * np.pi * t),
        'cos_t': np.cos(2 * np.pi * t),
    })
    y = df['tm_k'].values
    return X, y, df


def evaluate_model(name, model, X, y, df, label=""):
    """Evaluate and print comprehensive statistics."""
    y_pred = model.predict(X)

    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)
    bias = np.mean(y_pred - y)
    max_res = np.max(y_pred - y)
    min_res = np.min(y_pred - y)

    print(f"\n{'='*60}")
    print(f"EVALUATION: {name} — {label}")
    print(f"{'='*60}")
    print(f"R²:           {r2:.4f}")
    print(f"RMSE:         {rmse:.3f} K")
    print(f"MAE:          {mae:.3f} K")
    print(f"Bias:         {bias:.3f} K")
    print(f"Max residual: {max_res:.3f} K")
    print(f"Min residual: {min_res:.3f} K")
    print(f"Observations: {len(y):,}")

    # Residuals by station
    residuals = y_pred - y
    df_copy = df.copy()
    df_copy['residual'] = residuals
    df_copy['abs_residual'] = np.abs(residuals)
    df_copy['squared_residual'] = residuals ** 2

    print(f"\nResiduals by location:")
    stats = df_copy.groupby('point_name').agg({
        'squared_residual': lambda x: np.sqrt(np.mean(x)),
        'residual': 'mean',
        'abs_residual': 'mean',
        'time': 'count'
    }).round(3)
    stats.columns = ['RMSE', 'Bias', 'MAE', 'Count']
    print(stats.to_string())

    # Residuals by month
    print(f"\nResiduals by month:")
    monthly = df_copy.groupby('month').agg({
        'squared_residual': lambda x: np.sqrt(np.mean(x)),
        'residual': 'mean',
        'time': 'count'
    }).round(3)
    monthly.columns = ['RMSE', 'Bias', 'Count']
    print(monthly.to_string())

    return {
        'name': name,
        'r2': r2,
        'rmse': rmse,
        'mae': mae,
        'bias': bias,
        'max_res': max_res,
        'min_res': min_res,
        'pred': y_pred,
        'residuals': residuals
    }


def save_models(models, results, output_dir):
    """Save all models, coefficients, and comparison summary."""
    output_dir.mkdir(exist_ok=True)

    # Save each model
    for name, model in models.items():
        model_file = output_dir / f'nigeria_tm_model_{name}.pkl'
        with open(model_file, 'wb') as f:
            pickle.dump(model, f)
        print(f"\n[OK] Saved: {model_file}")

    # Save linear coefficients as JSON (for API use)
    lr = models['linear']
    coefs = {
        'a0': lr.intercept_,
        'a1': lr.coef_[0],   # ts
        'a2': lr.coef_[1],   # ps
        'a3': lr.coef_[2],   # rhs
        'a4': lr.coef_[3],   # h
        'a5': lr.coef_[4],   # sin_t
        'a6': lr.coef_[5],   # cos_t
    }
    coef_file = output_dir / 'nigeria_tm_coefficients.json'
    with open(coef_file, 'w') as f:
        json.dump({k: float(v) for k, v in coefs.items()}, f, indent=2)
    print(f"[OK] Saved: {coef_file}")

    # Print linear equation
    print(f"\nLinear Model Equation:")
    print(f"Tm = {coefs['a0']:.4f} + {coefs['a1']:.4f}*Ts + {coefs['a2']:.4f}*Ps + "
          f"{coefs['a3']:.4f}*RHs + {coefs['a4']:.4f}*h + "
          f"{coefs['a5']:.4f}*sin(2πt) + {coefs['a6']:.4f}*cos(2πt)")

    # Save model comparison summary
    comparison = {
        'models': [
            {
                'name': r['name'],
                'r2': round(r['r2'], 4),
                'rmse_k': round(r['rmse'], 3),
                'mae_k': round(r['mae'], 3),
                'bias_k': round(r['bias'], 3),
            }
            for r in results
        ],
        'best_model': min(results, key=lambda x: x['rmse'])['name'],
        'training_period': '2020-2023',
        'validation_period': '2024',
    }

    comp_file = output_dir / 'model_comparison.json'
    with open(comp_file, 'w') as f:
        json.dump(comparison, f, indent=2)
    print(f"[OK] Saved: {comp_file}")

    return coefs


def main():
    print("=" * 60)
    print("Nigeria GNSS Meteorology - ML Model Fitter")
    print("=" * 60)

    df = load_data()
    if df is None:
        return

    X, y, df = prepare_features(df)

    # Split: 2020-2023 for training, 2024 for validation
    train_mask = df['year'] < 2024
    test_mask = df['year'] == 2024

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    df_train = df[train_mask].copy()
    df_test = df[test_mask].copy()

    print(f"\nTraining:   {len(X_train):,} records (2020-2023)")
    print(f"Validation: {len(X_test):,} records (2024)")
    print(f"Features:   {list(X.columns)}")

    models = {}
    results = []

    # 1. Linear Regression (baseline — physically interpretable)
    print("\n[Fitting Linear Regression...]")
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    models['linear'] = lr
    results.append(evaluate_model("Linear Regression", lr, X_test, y_test, df_test, "Validation (2024)"))

    # 2. Random Forest (ensemble machine learning)
    print("\n[Fitting Random Forest (100 trees, max_depth=15)...]")
    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    models['random_forest'] = rf
    results.append(evaluate_model("Random Forest", rf, X_test, y_test, df_test, "Validation (2024)"))

    # 3. XGBoost (gradient boosting — if available)
    if XGBOOST_AVAILABLE:
        print("\n[Fitting XGBoost (100 estimators, max_depth=6)...]")
        xgb_model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            reg_alpha=0.1,
            reg_lambda=1.0
        )
        xgb_model.fit(X_train, y_train)
        models['xgboost'] = xgb_model
        results.append(evaluate_model("XGBoost", xgb_model, X_test, y_test, df_test, "Validation (2024)"))

    # Model comparison table
    print("\n" + "=" * 60)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 60)
    print(f"{'Model':<20} {'R²':>8} {'RMSE (K)':>10} {'MAE (K)':>10} {'Bias (K)':>10}")
    print("-" * 60)
    for r in results:
        marker = " ★" if r['name'] == min(results, key=lambda x: x['rmse'])['name'] else ""
        print(f"{r['name']:<20} {r['r2']:>8.4f} {r['rmse']:>10.3f} {r['mae']:>10.3f} {r['bias']:>10.3f}{marker}")
    print("-" * 60)

    best = min(results, key=lambda x: x['rmse'])
    print(f"\nBest model by RMSE: {best['name']} (RMSE = {best['rmse']:.3f} K)")

    # Feature importance for Random Forest
    if 'random_forest' in models:
        rf = models['random_forest']
        importances = pd.Series(rf.feature_importances_, index=X.columns)
        importances = importances.sort_values(ascending=False)
        print(f"\nRandom Forest Feature Importance:")
        for feat, imp in importances.items():
            print(f"  {feat:8s}: {imp:.4f} ({imp*100:.1f}%)")

    # Save everything
    BASE_DIR = Path(__file__).parent
    MODEL_DIR = BASE_DIR / 'models'
    save_models(models, results, MODEL_DIR)

    # Save predictions for all models
    df_test['tm_pred_linear'] = models['linear'].predict(X_test)
    df_test['tm_pred_rf'] = models['random_forest'].predict(X_test)
    if 'xgboost' in models:
        df_test['tm_pred_xgb'] = models['xgboost'].predict(X_test)

    pred_file = BASE_DIR / 'data' / 'processed' / 'validation_predictions_2024.csv'
    df_test.to_csv(pred_file, index=False)
    print(f"\n[OK] Predictions saved: {pred_file}")

    print("\n" + "=" * 60)
    print("ML MODEL FITTING COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()