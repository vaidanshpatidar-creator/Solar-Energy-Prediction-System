"""
Solar Energy Dataset Generator
Generates a realistic synthetic dataset for solar energy prediction.
Features: solar_radiation, temperature, cloud_cover, humidity,
          wind_speed, hour, day_of_week, month
Target:   energy_output (kWh)
"""

import numpy as np
import pandas as pd
import os

np.random.seed(42)

def generate_solar_dataset(n_samples: int = 10000, output_path: str = "data/solar_data.csv") -> pd.DataFrame:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # --- Time features ---
    months      = np.random.randint(1, 13, n_samples)          # 1–12
    days        = np.random.randint(1, 32, n_samples)           # 1–31
    hours       = np.random.randint(0, 24, n_samples)           # 0–23
    day_of_week = np.random.randint(0, 7, n_samples)            # 0=Mon … 6=Sun

    # --- Solar radiation (W/m²) —— peaks at solar noon, higher in summer ---
    # Hour factor: bell curve around noon (12)
    hour_factor = np.exp(-0.5 * ((hours - 12) / 3.5) ** 2)
    # Month factor: higher in summer months (Jun–Aug)
    month_factor = 0.5 + 0.5 * np.sin((months - 3) * np.pi / 6)
    solar_radiation = (
        1000 * hour_factor * month_factor
        + np.random.normal(0, 40, n_samples)
    ).clip(0, 1100)

    # --- Temperature (°C) ---
    temp_base = 15 + 10 * np.sin((months - 1) * np.pi / 6)          # seasonal
    temperature = (
        temp_base
        + 5 * np.sin((hours - 6) * np.pi / 12)                      # diurnal
        + np.random.normal(0, 3, n_samples)
    ).clip(-10, 50)

    # --- Cloud cover (%) ---
    cloud_cover = np.random.beta(2, 5, n_samples) * 100              # skewed low

    # --- Humidity (%) ---
    humidity = (
        50 + 20 * np.sin((months - 1) * np.pi / 6)                  # seasonal
        + np.random.normal(0, 10, n_samples)
    ).clip(10, 100)

    # --- Wind speed (m/s) ---
    wind_speed = np.random.weibull(2.0, n_samples) * 5               # Weibull dist

    # --- Energy output (kWh) — physics-inspired formula ---
    # Cloud cover reduces output significantly
    cloud_penalty = 1 - 0.8 * (cloud_cover / 100)
    # Temperature coefficient: panels lose efficiency above ~25°C
    temp_coeff = 1 - 0.004 * np.maximum(temperature - 25, 0)
    # Humidity slightly reduces output
    humidity_penalty = 1 - 0.002 * (humidity - 50).clip(0, None)

    energy_output = (
        solar_radiation * 0.02 * cloud_penalty * temp_coeff * humidity_penalty
        + np.random.normal(0, 0.5, n_samples)
    ).clip(0, None)

    # Night-time: near-zero output
    night_mask = (hours < 6) | (hours > 20)
    energy_output[night_mask] *= 0.02

    df = pd.DataFrame({
        "month":            months,
        "day":              days,
        "hour":             hours,
        "day_of_week":      day_of_week,
        "solar_radiation":  np.round(solar_radiation, 2),
        "temperature":      np.round(temperature, 2),
        "cloud_cover":      np.round(cloud_cover, 2),
        "humidity":         np.round(humidity, 2),
        "wind_speed":       np.round(wind_speed, 2),
        "energy_output":    np.round(energy_output, 3),
    })

    df.to_csv(output_path, index=False)
    print(f"[OK] Dataset saved -> {output_path}  ({len(df):,} rows)")
    return df


if __name__ == "__main__":
    df = generate_solar_dataset()
    print(df.describe())
