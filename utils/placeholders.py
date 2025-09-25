# utils/placeholders.py
import pandas as pd
import io
import base64
import time
import numpy as np

def run_revenue_model_placeholder(params, input_df, progress_callback):
    """
    A corrected placeholder that now uses the data from the uploaded file
    to generate plausible results.
    """
    progress_callback("Reading data...")
    time.sleep(1) # Simulate initial setup

    df = input_df.copy()
    df['Datetime'] = pd.to_datetime(df['Datetime'], dayfirst=True)
    df = df.set_index('Datetime')

    progress_callback("Running simulation...")
    time.sleep(2) # Simulate a long calculation

    # --- THIS IS THE CRITICAL FIX FOR INCORRECT RESULTS ---
    # We now use the actual uploaded data to calculate a result.
    # This is a simple example: revenue is positive when exporting (net < 0) and negative when importing.
    net_load = df['load'] - df.get('production_PV', 0)
    
    # Simulate a price that is high when net load is high (import)
    price = 50 + (net_load / net_load.max()) * 100
    
    # revenue = -net_load * price (negative net_load is export/income)
    df['total_result_day_ahead_trading'] = -net_load * price / 1000 # Convert to MWh for price

    if 'production_PV' not in df.columns:
        df['production_PV'] = 0
    if 'load' not in df.columns:
        df['load'] = 0
        
    df['SoC_kWh'] = (df['total_result_day_ahead_trading'].cumsum() / 100).clip(0, params.get("CAPACITY_MWH", 2) * 1000)

    summary = {
        'optimization_method': 'Placeholder Heuristic Model',
        'total_cycles': np.random.uniform(10, 20),
        'infeasible_days': []
    }
    
    output_stream = io.BytesIO()
    with pd.ExcelWriter(output_stream, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Results')
    output_stream.seek(0)
    
    file_bytes_b64 = base64.b64encode(output_stream.getvalue()).decode('utf-8')

    return {
        "df": df.to_json(orient='split'),
        "summary": summary,
        "output_file_bytes_b64": file_bytes_b64,
        "warnings": ["Warning: Using placeholder data. Results are calculated from your inputs but are not from the full optimization model."],
        "error": None
    }

# The battery shaving placeholder remains the same as it was already correct.
def run_battery_shaving_placeholder(input_df, import_limit, export_limit):
    time.sleep(1)
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