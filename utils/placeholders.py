# utils/placeholders.py
import pandas as pd
from datetime import datetime, timedelta

def run_revenue_model_placeholder(params, input_df, progress_callback):
    """A placeholder that mimics the output of the real revenue model."""
    progress_callback("Running placeholder revenue model...")

    # Create a sample results DataFrame based on the input
    df = input_df.copy()
    df['Datetime'] = pd.to_datetime(df['Datetime'], dayfirst=True)
    df = df.set_index('Datetime')

    df['total_result_day_ahead_trading'] = 150 * (pd.Series(df.index.hour) - 12) * -1
    if 'production_PV' not in df.columns:
        df['production_PV'] = 1000 * (1 - (pd.Series(df.index.hour) - 14).abs() / 8).clip(0)
    if 'load' not in df.columns:
        df['load'] = 500 + 200 * pd.Series(df.index.hour) / 24
    df['SoC_kWh'] = 1000 + 500 * (1 - (pd.Series(df.index.hour) - 16).abs() / 10).clip(0)

    # Create sample summary and file bytes
    summary = {
        'optimization_method': 'Placeholder Linear Program',
        'total_cycles': 12.5,
        'infeasible_days': ['2025-01-15', '2025-07-22']
    }
    # Create a dummy Excel file in memory to simulate download
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Results')
    output.seek(0)

    return {
        "df": df.to_json(orient='split'), # Return as JSON
        "summary": summary,
        "output_file_bytes": output.getvalue(),
        "warnings": ["Warning: Using placeholder data. This is not a real simulation."],
        "error": None
    }

def run_battery_shaving_placeholder(input_df, import_limit, export_limit):
    """A placeholder that mimics the output of the battery shaving analyzer."""
    df = input_df.copy()
    df["Datetime"] = pd.to_datetime(df["Datetime"], dayfirst=True)
    df.set_index("Datetime", inplace=True)

    df['net_load'] = df['load'] - df['pv_production']
    df['battery_power'] = - (df['net_load'] - (import_limit + export_limit)/2).clip(lower=(df['net_load'] - import_limit) * -1, upper=(df['net_load'] - export_limit) * -1)
    df['soc_kwh'] = (df['battery_power'] * 0.25).cumsum() # Assume 15-min intervals
    df['soc_kwh'] -= df['soc_kwh'].min() # Normalize to start at 0

    capacity = df['soc_kwh'].max()
    power = df['battery_power'].abs().max()

    return capacity, power, df.to_json(orient='split') # Return as JSON