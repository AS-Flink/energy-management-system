# pages/revenue_analysis.py
import dash
from dash import html, dcc, callback, Input, Output, State, no_update
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import base64
from datetime import datetime

from utils.placeholders import run_revenue_model_placeholder
from utils.data_processing import parse_contents, find_total_result_column, resample_data
from utils.diagrams import create_horizontal_diagram_with_icons

dash.register_page(__name__, path='/revenue-analysis', name='Revenue Analysis 💰')

# --- Reusable Components & Data ---
def create_metric(label, value_id, default_value="--"):
    return dbc.Card(dbc.CardBody([
        html.H6(label, className="card-title text-muted"),
        html.H4(default_value, className="card-text", id=value_id)
    ]))

situation_options = [
    "Situation 1: PV + Consumption on PAP", "Situation 2: PV on SAP, Consumption on PAP",
    "Situation 3: PV+Consumption on PAP, Battery on SAP", "Situation 4: Everything on PAP (Imbalance)",
    "Situation 5: Consumption on PAP, Battery+PV on SAP",
    "Situation 6: Consumption on PAP, Battery on SAP1, PV on SAP2", "Situation 7: PV + Battery on PAP"
]

# --- Sidebar Layout ---
sidebar = dbc.Card(dbc.CardBody([
    html.H4("⚙️ Configuration", className="mb-3"),
    dbc.Label("1. System Configuration"),
    dcc.Dropdown(id="ra-situation-dropdown", options=situation_options, value=situation_options[3]),
    html.Hr(),
    dbc.Label("2. Upload Data (CSV or Excel)"),
    dcc.Upload(
        id='ra-upload-data',
        children=html.Div(['Drag and Drop or ', html.A('Select Files')]),
        style={'borderWidth': '1px', 'borderStyle': 'dashed', 'borderRadius': '5px', 'textAlign': 'center', 'padding': '10px'},
    ),
    html.Div(id='ra-upload-status', className="mt-2 small"),
    html.Hr(),
    dbc.Label("3. Optimization Strategy"),
    # ... Add all other inputs from your Streamlit app here ...
    dcc.Dropdown(id='ra-strategy-dropdown', options=["Prioritize Self-Consumption", "Optimize on Day-Ahead Market", "Simple Battery Trading (Imbalance)"], value="Simple Battery Trading (Imbalance)"),
    html.Hr(),
    dbc.Button("🚀 Run Analysis", id="ra-run-button", color="primary", n_clicks=0, className="w-100"),
]), className="h-100")

# --- Main Content Layout ---
main_content = dbc.Container([
    html.H1("Revenue Analysis Simulation"),
    html.P("Configure your simulation, upload data, and run the analysis."),
    html.Hr(),
    html.H4("Selected Configuration"),
    html.Div(id='ra-diagram-output', className="mb-4"),
    html.Hr(),
    dcc.Loading(id="ra-loading-spinner", children=html.Div(id="ra-results-output"))
], fluid=True)

# --- Full Page Layout ---
layout = dbc.Container([
    dbc.Row([
        dbc.Col(sidebar, width=12, lg=3, className="mb-4"),
        dbc.Col(main_content, width=12, lg=9),
    ]),
    # Hidden stores
    dcc.Store(id='ra-results-store'),
    dcc.Store(id='ra-input-df-store'),
], fluid=True)

# --- Callbacks ---

@callback(
    Output('ra-diagram-output', 'children'),
    Input('ra-situation-dropdown', 'value')
)
def update_diagram(situation):
    return create_horizontal_diagram_with_icons(situation)

@callback(
    [Output('ra-upload-status', 'children'), Output('ra-input-df-store', 'data')],
    Input('ra-upload-data', 'contents'),
    State('ra-upload-data', 'filename'),
    prevent_initial_call=True
)
def handle_upload(contents, filename):
    if contents is None:
        return no_update, no_update
    
    # Add 'revenue' to filename to trigger specific sheet name parsing
    filename_for_parsing = f"revenue_{filename}"
    df, message = parse_contents(contents, filename_for_parsing)

    if df is None:
        return html.Div(message, className="text-danger"), None
    
    return html.Div(message, className="text-success"), df.to_json(date_format='iso', orient='split')

@callback(
    Output('ra-results-store', 'data'),
    Input('ra-run-button', 'n_clicks'),
    [State('ra-input-df-store', 'data'), State('ra-strategy-dropdown', 'value')],
    prevent_initial_call=True
)
def run_model(n_clicks, df_json, strategy):
    if df_json is None:
        return {"error": "Please upload a data file first."}
    
    input_df = pd.read_json(df_json, orient='split')
    params = {"STRATEGY_CHOICE": strategy}
    
    # Using the placeholder
    results = run_revenue_model_placeholder(params, input_df, lambda msg: print(msg))
    return results

@callback(
    Output('ra-results-output', 'children'),
    Input('ra-results-store', 'data')
)
def display_results(results_data):
    if not results_data:
        return dbc.Alert("Configure and run the analysis to see results.", color="info")
    if results_data.get("error"):
        return dbc.Alert(f"Error: {results_data['error']}", color="danger")
    
    summary = results_data["summary"]
    df = pd.read_json(results_data["df"], orient='split')
    total_result_col = find_total_result_column(df)
    net_result = df[total_result_col].sum() if total_result_col and not df.empty else 0
    
    download_button = dcc.Download(id="ra-download-excel")

    return html.Div([
        download_button,
        dbc.Row([
            dbc.Col(create_metric("Net Result / Revenue", 'ra-metric-net-result', f"€ {net_result:,.0f}")),
            dbc.Col(create_metric("Total Cycles", 'ra-metric-cycles', f"{summary.get('total_cycles', 0):.1f}")),
            dbc.Col(create_metric("Infeasible Days", 'ra-metric-infeasible', f"{len(summary.get('infeasible_days', []))}")),
        ]),
        dbc.Button("📥 Download Full Results (Excel)", id="ra-download-button", color="secondary", className="mt-3 mb-3"),
        html.Hr(),
        html.H4("📊 Interactive Charts"),
        # ... You can add a resolution selector and tabs here just like in the Streamlit app ...
        dcc.Graph(figure=px.line(df, y=total_result_col, title="Financial Result"))
    ])

@callback(
    Output("ra-download-excel", "data"),
    Input("ra-download-button", "n_clicks"),
    State("ra-results-store", "data"),
    prevent_initial_call=True,
)
def download_excel(n_clicks, results_data):
    if not results_data or "output_file_bytes" not in results_data:
        return no_update
    
    # The placeholder creates bytes, but in a real scenario you would decode them
    # For this placeholder, we can just pass the bytes through.
    file_bytes_b64 = base64.b64encode(results_data["output_file_bytes"]).decode()
    
    return dict(
        content=file_bytes_b64,
        filename=f"Revenue_Analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
        base64=True
    )