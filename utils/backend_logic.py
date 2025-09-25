# utils/backend_logic.py
import sys
import os
import pandas as pd
import io
import base64
import time

# --- THIS IS THE FIX ---
# This code tells Python to look in the main project folder for modules.
# It makes the import `from main_models...` work correctly.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)
# --------------------

# Now, Python can find your 'main_models' folder
from main_models.self_consumption_PV_PAP import run_simulation as run_self_consumption_model
from main_models.day_ahead_trading_PAP import run_simulation as run_day_ahead_model
from main_models.imbalance_everything_PAP import run_simulation as run_imbalance_pap_model
from main_models.imbalance_algorithm_SAP import run_simulation as run_imbalance_sap_model

def run_master_simulation(params, input_df, progress_callback):
    """
    This master function calls the correct model based on the user's strategy choice.
    """
    strategy = params.get("STRATEGY_CHOICE")
    
    model_mapping = {
        "Prioritize Self-Consumption": run_self_consumption_model,
        "Optimize on Day-Ahead Market": run_day_ahead_model,
        "Advanced Whole-System Trading (Imbalance)": run_imbalance_pap_model,
        "Simple Battery Trading (Imbalance)": run_imbalance_sap_model
    }

    model_to_run = model_mapping.get(strategy)

    if not model_to_run:
        return {"error": f"Strategy '{strategy}' is not yet connected to a model script."}

    try:
        results = model_to_run(params, input_df, progress_callback)
    except Exception as e:
        return {"error": f"An error occurred while running the simulation: {e}"}

    if results.get("error"):
        return results

    df = results.get('df')
    if df is None:
        return {"error": "The simulation ran but did not return a DataFrame."}
        
    output_stream = io.BytesIO()
    with pd.ExcelWriter(output_stream, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Results')
    output_stream.seek(0)
    
    file_bytes_b64 = base64.b64encode(output_stream.getvalue()).decode('utf-8')

    results['df'] = df.to_json(orient='split')
    results['output_file_bytes_b64'] = file_bytes_b64
    
    return results