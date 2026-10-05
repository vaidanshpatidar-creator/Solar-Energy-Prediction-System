"""
Train a Random Forest model for solar energy output prediction.
Saves the trained model and a feature-importance CSV to the models/ directory.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# ── paths ────────────────────────────────────────────────────────────────────
DATA_PATH        = "data/solar_data.csv"
MODEL_DIR        = "models"
MODEL_PATH       = os.path.join(MODEL_DIR, "rf_solar_model.pkl")
SCALER_PATH      = os.path.join(MODEL_DIR, "scaler.pkl")
METRICS_PATH     = os.path.join(MODEL_DIR, "metrics.json")
IMPORTANCE_PATH  = os.path.join(MODEL_DIR, "feature_importance.csv")

FEATURE_COLS = [
    "month", "day", "hour", "day_of_week",
    "solar_radiation", "temperature",
    "cloud_cover", "humidity", "wind_speed",
]
TARGET_COL = "energy_output"


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        print("Dataset not found – generating now …")
        from generate_dataset import generate_solar_dataset
        generate_solar_dataset(output_path=path)
    return pd.read_csv(path)


def train(df: pd.DataFrame):
    os.makedirs(MODEL_DIR, exist_ok=True)

    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale features (Random Forest doesn't require it, but helps downstream)
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    # ── Random Forest ────────────────────────────────────────────────────────
    rf = RandomForestRegressor(
        n_estimators     = 200,
        max_depth        = 20,
        min_samples_leaf = 2,
        min_samples_split= 4,
        max_features     = "sqrt",
        n_jobs           = -1,
        random_state     = 42,
    )
    rf.fit(X_train_sc, y_train)

    # ── Evaluation ───────────────────────────────────────────────────────────
    y_pred = rf.predict(X_test_sc)
    mae  = float(np.round(mean_absolute_error(y_test, y_pred), 4))
    rmse = float(np.round(np.sqrt(mean_squared_error(y_test, y_pred)), 4))
    r2   = float(np.round(r2_score(y_test, y_pred), 4))
    cv_r2 = float(np.round(
        cross_val_score(rf, scaler.transform(X), y, cv=5, scoring="r2").mean(), 4
    ))

    metrics = {"MAE": mae, "RMSE": rmse, "R2": r2, "CV_R2": cv_r2,
               "train_samples": len(X_train), "test_samples": len(X_test)}

    # ── Feature importance ───────────────────────────────────────────────────
    fi = pd.DataFrame({
        "feature":   FEATURE_COLS,
        "importance": rf.feature_importances_,
    }).sort_values("importance", ascending=False)

    # ── Persist ──────────────────────────────────────────────────────────────
    joblib.dump(rf,     MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    fi.to_csv(IMPORTANCE_PATH, index=False)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n[Model Performance]")
    print(f"   MAE   : {mae}")
    print(f"   RMSE  : {rmse}")
    print(f"   R2    : {r2}")
    print(f"   CV R2 : {cv_r2}")
    print(f"\n[OK] Model saved  -> {MODEL_PATH}")
    print(f"[OK] Scaler saved -> {SCALER_PATH}")

    return rf, scaler, metrics, fi


if __name__ == "__main__":
    df = load_data()
    train(df)
