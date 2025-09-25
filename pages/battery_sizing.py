# pages/battery_sizing.py
import dash
from dash import html, dcc, callback, Input, Output, State, no_update
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from utils.placeholders import run_battery_shaving_placeholder
from utils.data_processing import parse_contents

dash.register_page(__name__, path='/battery-sizing', name='Battery Sizing 🛠️')

# --- Layout ---
sidebar = dbc.Card(dbc.CardBody([
    html.H4("⚙️ Sizing Configuration"),
    html.Hr(),
    dbc.Label("Upload Your Data (CSV or Excel)"),
    dcc.Upload(
        id='bs-upload-data',
        children=html.Div(['Drag and Drop or ', html.A('Select Files')]),
        style={'borderWidth': '1px', 'borderStyle': 'dashed', 'borderRadius': '5px', 'textAlign': 'center', 'padding': '10px'},
    ),
    html.Div(id='bs-upload-status', className="mt-2 small"),
    html.Hr(),
    dbc.Label("Grid Import Limit (kW)"),
    dbc.Input(id='bs-import-limit', type='number', value=350, step=10),
    html.Br(),
    dbc.Label("Grid Export Limit (kW)"),
    dbc.Input(id='bs-export-limit', type='number', value=-250, step=10),
    html.Hr(),
    dbc.Button("🚀 Run Sizing Analysis", id="bs-run-button", color="primary", n_clicks=0, className="w-100"),
]), className="h-100")

main_content = dbc.Container([
    html.H1("Battery Sizing Tool for Peak Shaving"),
    dbc.Alert("This tool calculates the battery power (kW) and capacity (kWh) needed to keep your grid exchange within defined limits.", color="info"),
    html.Hr(),
    dcc.Loading(id="bs-loading-spinner", children=html.Div(id="bs-results-output"))
], fluid=True)

layout = dbc.Container([
    dbc.Row([
        dbc.Col(sidebar, width=12, lg=3, className="mb-4"),
        dbc.Col(main_content, width=12, lg=9),
    ]),
    dcc.Store(id='bs-results-store'),
    dcc.Store(id='bs-input-df-store'),
], fluid=True)

# --- Callbacks ---

@callback(
    [Output('bs-upload-status', 'children'), Output('bs-input-df-store', 'data')],
    Input('bs-upload-data', 'contents'),
    State('bs-upload-data', 'filename'),
    prevent_initial_call=True
)
def handle_upload_bs(contents, filename):
    if contents is None:
        return no_update, no_update
    
    df, message = parse_contents(contents, filename)
    
    # Additional validation for this specific tool
    if df is not None and not all(col in df.columns for col in ['load', 'pv_production']):
        message = "Error: CSV/Excel must have 'load' and 'pv_production' columns."
        return html.Div(message, className="text-danger"), None

    if df is None:
        return html.Div(message, className="text-danger"), None
    
    return html.Div(message, className="text-success"), df.to_json(date_format='iso', orient='split')

@callback(
    Output('bs-results-store', 'data'),
    Input('bs-run-button', 'n_clicks'),
    [State('bs-input-df-store', 'data'), State('bs-import-limit', 'value'), State('bs-export-limit', 'value')],
    prevent_initial_call=True
)
def run_sizing(n_clicks, df_json, import_limit, export_limit):
    if df_json is None:
        return {"error": "Please upload a data file first."}
    
    input_df = pd.read_json(df_json, orient='split')
    capacity, power, results_df_json = run_battery_shaving_placeholder(input_df, import_limit, export_limit)
    
    return {
        "capacity": capacity,
        "power": power,
        "df": results_df_json,
        "import_limit": import_limit,
        "export_limit": export_limit,
        "error": None
    }

@callback(
    Output('bs-results-output', 'children'),
    Input('bs-results-store', 'data')
)
def display_sizing_results(results_data):
    if not results_data:
        return dbc.Alert("Upload a file and run the analysis to see results.", color="info")
    if results_data.get("error"):
        return dbc.Alert(f"Error: {results_data['error']}", color="danger")
    
    df = pd.read_json(results_data['df'], orient='split')
    
    # Plot 1: Net Load vs Limits
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=df.index, y=df['net_load'], mode='lines', name='Net Load', line=dict(color='royalblue', width=1)))
    fig1.add_hline(y=results_data['import_limit'], line_dash="dash", line_color="red", annotation_text=f"Import Limit ({results_data['import_limit']} kW)")
    fig1.add_hline(y=results_data['export_limit'], line_dash="dash", line_color="green", annotation_text=f"Export Limit ({results_data['export_limit']} kW)")
    fig1.update_layout(title="Net Load vs. Grid Limits", xaxis_title="Time", yaxis_title="Power (kW)")

    # Plot 2: Battery Power
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=df.index, y=df['battery_power'].clip(lower=0), mode='lines', name='Charging Power', fill='tozeroy', line=dict(color='green')))
    fig2.add_trace(go.Scatter(x=df.index, y=df['battery_power'].clip(upper=0), mode='lines', name='Discharging Power', fill='tozeroy', line=dict(color='red')))
    fig2.update_layout(title="Required Battery Power to Shave Peaks", xaxis_title="Time", yaxis_title="Power (kW)")
    
    # Plot 3: Battery SoC
    fig3 = px.line(df, y='soc_kwh', title="Calculated Battery State of Charge", labels={"value": "Energy (kWh)", "index": "Time"})

    return html.Div([
        html.H4("💡 Recommended Battery Size"),
        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([html.H6("Required Power"), html.H4(f"{results_data['power']:,.2f} kW")]))),
            dbc.Col(dbc.Card(dbc.CardBody([html.H6("Required Capacity"), html.H4(f"{results_data['capacity']:,.2f} kWh")]))),
        ]),
        html.Hr(),
        html.H4("📊 Analysis Charts"),
        dcc.Graph(figure=fig1),
        dcc.Graph(figure=fig2),
        dcc.Graph(figure=fig3)
    ])