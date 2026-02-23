"""
TurboForge - Data Preparation Script
Converts raw Kaggle T1.csv into scada_real.csv (50-turbine fleet)

Download T1.csv from:
https://www.kaggle.com/datasets/berkerisen/wind-turbine-scada-dataset

Then run:
    python data/prepare_data.py --input T1.csv --output scada_real.csv
"""

import argparse
import numpy as np
import pandas as pd


def prepare_data(input_path: str, output_path: str, n_turbines: int = 50):
    print(f"[Data] Loading {input_path}...")
    df = pd.read_csv(input_path)
    df.columns = ['timestamp', 'power_output_kw', 'wind_speed_ms',
                  'theoretical_power_kwh', 'wind_direction_deg']
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%d %m %Y %H:%M')
    df['power_output_kw'] = df['power_output_kw'].clip(0, 3000)
    df = df.sort_values('timestamp').reset_index(drop=True)

    # Failure label: actual power < 50% of theoretical when wind is sufficient
    df['failure_label'] = (
        (df['theoretical_power_kwh'] > 100) &
        (df['power_output_kw'] < df['theoretical_power_kwh'] * 0.5)
    ).astype(int)

    print(f"[Data] Base turbine: {len(df):,} records | Failure rate: {df['failure_label'].mean():.2%}")

    # Derive sensor features from available columns
    np.random.seed(42)
    n = len(df)
    df['rotor_rpm']        = (df['wind_speed_ms'] * 1.8 + np.random.normal(0, 0.3, n)).clip(0, 20)
    df['blade_pitch_deg']  = (10 - df['wind_speed_ms'] * 0.3 + np.random.normal(0, 0.5, n)).clip(0, 90)
    df['nacelle_temp_c']   = 25 + df['power_output_kw'] * 0.005 + np.random.normal(0, 1.5, n)
    df['gearbox_temp_c']   = 40 + df['power_output_kw'] * 0.009 + np.random.normal(0, 2, n)
    df['generator_temp_c'] = 55 + df['power_output_kw'] * 0.012 + np.random.normal(0, 2, n)
    df['vibration_x']      = np.abs(np.random.normal(0.02, 0.005, n))
    df['vibration_y']      = np.abs(np.random.normal(0.02, 0.005, n))
    df['turbine_id'] = 1

    # Simulate n_turbines with realistic per-turbine variance
    all_dfs = [df.copy()]
    for tid in range(2, n_turbines + 1):
        dt = df.copy()
        dt['turbine_id'] = tid
        wind_off   = np.random.uniform(-1.5, 1.5)
        pwr_factor = np.random.uniform(0.92, 1.08)
        temp_off   = np.random.uniform(-3, 5)
        wear       = np.random.uniform(0.95, 1.1)

        dt['wind_speed_ms']    = (dt['wind_speed_ms'] + wind_off).clip(0, 25)
        dt['power_output_kw']  = (dt['power_output_kw'] * pwr_factor).clip(0, 3000)
        dt['rotor_rpm']        = (dt['wind_speed_ms'] * 1.8 + np.random.normal(0, 0.3, n)).clip(0, 20)
        dt['blade_pitch_deg']  = (10 - dt['wind_speed_ms'] * 0.3 + np.random.normal(0, 0.5, n)).clip(0, 90)
        dt['nacelle_temp_c']   = 25 + dt['power_output_kw'] * 0.005 + temp_off + np.random.normal(0, 1.5, n)
        dt['gearbox_temp_c']   = 40 + dt['power_output_kw'] * 0.009 * wear + temp_off + np.random.normal(0, 2, n)
        dt['generator_temp_c'] = 55 + dt['power_output_kw'] * 0.012 + temp_off * 0.8 + np.random.normal(0, 2, n)
        dt['vibration_x']      = np.abs(np.random.normal(0.02 * wear, 0.005, n))
        dt['vibration_y']      = np.abs(np.random.normal(0.02 * wear, 0.005, n))

        # Aging turbines have higher failure rates
        if tid % 12 == 0:
            dt['gearbox_temp_c'] += 10
            dt['vibration_x']    *= 2.0
            dt.loc[np.random.rand(n) < 0.015, 'failure_label'] = 1

        all_dfs.append(dt)

    final = pd.concat(all_dfs, ignore_index=True)
    cols = ['timestamp', 'turbine_id', 'wind_speed_ms', 'rotor_rpm', 'power_output_kw',
            'blade_pitch_deg', 'nacelle_temp_c', 'gearbox_temp_c', 'generator_temp_c',
            'vibration_x', 'vibration_y', 'failure_label']
    final = final[cols]
    final.to_csv(output_path, index=False)

    print(f"[Data] Saved: {output_path}")
    print(f"[Data] {final['turbine_id'].nunique()} turbines | {len(final):,} records | Failure rate: {final['failure_label'].mean():.2%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="T1.csv")
    parser.add_argument("--output", default="scada_real.csv")
    parser.add_argument("--n_turbines", type=int, default=50)
    args = parser.parse_args()
    prepare_data(args.input, args.output, args.n_turbines)
