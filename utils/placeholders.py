# utils/placeholders.py
import pandas as pd
import io
import base64
import time

def run_revenue_model_placeholder(params, input_df, progress_callback):
    """A placeholder that mimics the output of the real revenue model."""
    progress_callback("Running placeholder revenue model...")
    time.sleep(2) # Simulate a long-running process

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

    summary = {
        'optimization_method': 'Placeholder Linear Program',
        'total_cycles': 12.5,
        'infeasible_days': ['2025-01-15', '2025-07-22']
    }
    
    # Create a dummy Excel file in memory
    output_stream = io.BytesIO()
    with pd.ExcelWriter(output_stream, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Results')
    output_stream.seek(0)
    
    # **THIS IS THE CRITICAL FIX**
    # We encode the raw bytes into a Base64 text string, which is JSON serializable.
    file_bytes_b64 = base64.b64encode(output_stream.getvalue()).decode('utf-8')

    return {
        "df": df.to_json(orient='split'),
        "summary": summary,
        "output_file_bytes_b64": file_bytes_b64, # Use the new encoded string
        "warnings": ["Warning: Using placeholder data. This is not a real simulation."],
        "error": None
    }

def run_battery_shaving_placeholder(input_df, import_limit, export_limit):
    """A placeholder that mimics the output of the BatteryShavingAnalyzer."""
    time.sleep(1) # Simulate a long-running process
    df = input_df.copy()
    df["Datetime"] = pd.to_datetime(df["Datetime"], dayfirst=True)
    df.set_index("Datetime", inplace=True)

    if 'load' not in df.columns: df['load'] = 0
    if 'pv_production' not in df.columns: df['pv_production'] = 0
    
    df['net_load'] = df['load'] - df['pv_production']
    df['battery_power'] = - (df['net_load'] - (import_limit + export_limit)/2).clip(
        lower=(df['net_load'] - import_limit) * -1, 
        upper=(df['net_load'] - export_limit) * -1
    )
    df['soc_kwh'] = (df['battery_power'] * 0.25).cumsum()
    if not df['soc_kwh'].empty:
        df['soc_kwh'] -= df['soc_kwh'].min() 

    capacity = df['soc_kwh'].max()
    power = df['battery_power'].abs().max()

    return capacity, power, df.to_json(orient='split')