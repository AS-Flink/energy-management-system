import pandas as pd
import io
import base64

def run_revenue_model_placeholder(params, input_df, progress_callback):
    """
    A placeholder that mimics the output of the real revenue model.
    It takes the uploaded dataframe and generates sample results.
    """
    progress_callback("Running placeholder revenue model...")

    # Create a sample results DataFrame based on the input
    df = input_df.copy()
    df['Datetime'] = pd.to_datetime(df['Datetime'], dayfirst=True)
    df = df.set_index('Datetime')

    # Generate some plausible-looking data for demonstration
    df['total_result_day_ahead_trading'] = 150 * (pd.Series(df.index.hour) - 12) * -1
    if 'production_PV' not in df.columns:
        df['production_PV'] = 1000 * (1 - (pd.Series(df.index.hour) - 14).abs() / 8).clip(0)
    if 'load' not in df.columns:
        df['load'] = 500 + 200 * pd.Series(df.index.hour) / 24
    df['SoC_kWh'] = 1000 + 500 * (1 - (pd.Series(df.index.hour) - 16).abs() / 10).clip(0)

    # Create a sample summary dictionary
    summary = {
        'optimization_method': 'Placeholder Linear Program',
        'total_cycles': 12.5,
        'infeasible_days': ['2025-01-15', '2025-07-22']
    }
    
    # Create a dummy Excel file in memory to simulate the download feature
    output_stream = io.BytesIO()
    with pd.ExcelWriter(output_stream, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Results')
    output_stream.seek(0)

    return {
        "df": df.to_json(orient='split'), # Return DataFrame as JSON
        "summary": summary,
        "output_file_bytes": output_stream.getvalue(),
        "warnings": ["Warning: Using placeholder data. This is not a real simulation."],
        "error": None
    }

def run_battery_shaving_placeholder(input_df, import_limit, export_limit):
    """
    A placeholder that mimics the output of the BatteryShavingAnalyzer.
    """
    df = input_df.copy()
    df["Datetime"] = pd.to_datetime(df["Datetime"], dayfirst=True)
    df.set_index("Datetime", inplace=True)

    # Ensure required columns exist, even if they are zero
    if 'load' not in df.columns: df['load'] = 0
    if 'pv_production' not in df.columns: df['pv_production'] = 0
    
    # Generate plausible data for demonstration
    df['net_load'] = df['load'] - df['pv_production']
    df['battery_power'] = - (df['net_load'] - (import_limit + export_limit)/2).clip(
        lower=(df['net_load'] - import_limit) * -1, 
        upper=(df['net_load'] - export_limit) * -1
    )
    # Assume 15-minute intervals for SoC calculation
    df['soc_kwh'] = (df['battery_power'] * 0.25).cumsum()
    # Normalize SoC to start at 0 to represent relative charge
    if not df['soc_kwh'].empty:
        df['soc_kwh'] -= df['soc_kwh'].min() 

    capacity = df['soc_kwh'].max()
    power = df['battery_power'].abs().max()

    return capacity, power, df.to_json(orient='split') # Return DataFrame as JSON