# utils/data_processing.py
import pandas as pd
import io
import base64

def find_total_result_column(df):
    """Finds the correct total result column in the DataFrame."""
    possible_cols = [
        'total_result_imbalance_PAP',
        'total_result_imbalance_SAP',
        'total_result_day_ahead_trading',
        'total_result_self_consumption'
    ]
    for col in possible_cols:
        if col in df.columns:
            return col
    return None

def resample_data(df, resolution):
    """Resamples the DataFrame to the specified time resolution."""
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            # Return an empty DataFrame or handle error as needed
            return pd.DataFrame()

    resolution_map = {
        'Hourly': 'H',
        'Daily': 'D',
        'Monthly': 'M',
        'Yearly': 'Y'
    }
    if resolution == '15 Min (Original)':
        return df

    rule = resolution_map.get(resolution)
    if rule:
        return df.resample(rule).sum()

    return df

def parse_contents(contents, filename):
    """Parses the content of an uploaded file (CSV or Excel)."""
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            # For the revenue analysis tool that uses a specific sheet name
            if 'revenue' in filename.lower():
                 df = pd.read_excel(io.BytesIO(decoded), sheet_name='Export naar Python', header=0)
            else: # For other tools like battery sizing
                 df = pd.read_excel(io.BytesIO(decoded))

        # Basic validation
        if 'Datetime' not in df.columns:
            return None, "Error: 'Datetime' column not found in the file."

        return df, f"✅ Loaded: {filename}"
    except Exception as e:
        return None, f"Error processing file: {e}. Check format and sheet name ('Export naar Python' for revenue analysis)."