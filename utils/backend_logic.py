# utils/backend_logic.py
import pandas as pd
import io
import base64

# ============================================================================
# IMPORTANT: This is where you will import the functions from your model files
# I am assuming your files are in a folder named 'main-models' and that each
# file contains a main function, for example, run_simulation().
# You may need to adjust the function names to match what's in your scripts.
# ============================================================================
from main_models.self_consumption_PV_PAP import run_simulation as run_self_consumption_model
from main_models.day_ahead_trading_PAP import run_simulation as run_day_ahead_model
from main_models.imbalance_everything_PAP import run_simulation as run_imbalance_pap_model
from main_models.imbalance_algorithm_SAP import run_simulation as run_imbalance_sap_model


def run_master_simulation(params, input_df, progress_callback):
    """
    This master function calls the correct model based on the user's strategy choice
    and passes the progress_callback down to it.
    """
    strategy = params.get("STRATEGY_CHOICE")
    
    # This dictionary maps the user's dropdown choice to your actual model functions.
    # This is based on the four strategies available in your Streamlit app.
    model_mapping = {
        "Prioritize Self-Consumption": run_self_consumption_model,
        "Optimize on Day-Ahead Market": run_day_ahead_model,
        "Advanced Whole-System Trading (Imbalance)": run_imbalance_pap_model,
        "Simple Battery Trading (Imbalance)": run_imbalance_sap_model
    }

    model_to_run = model_mapping.get(strategy)

    if not model_to_run:
        return {"error": f"Strategy '{strategy}' is not connected to a model script."}

    try:
        # We call your actual model, passing the parameters, the data, and the
        # progress_callback function so it can send live updates to the UI.
        results = model_to_run(params, input_df, progress_callback)
    except Exception as e:
        return {"error": f"An error occurred while running the simulation: {e}"}


    # --- Final Data Formatting ---
    # This part should be common for all models. It ensures the final output
    # is ready for the Dash frontend (DataFrame to JSON, file to Base64).
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