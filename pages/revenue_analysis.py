# pages/revenue_analysis.py
import dash
from dash import html, dcc, callback, Input, Output, State, no_update
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import base64
from datetime import datetime

# Import all necessary functions from your utils modules
from utils.placeholders import run_revenue_model_placeholder
from utils.data_processing import parse_contents, find_total_result_column, resample_data
from utils.diagrams import create_horizontal_diagram_with_icons

# --- Page Registration ---
dash.register_page(__name__, path='/revenue-analysis', name='Revenue Analysis 💰')

# --- Static Data ---
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
    dcc.Dropdown(id="ra-situation-dropdown", options=situation_options, value=situation_options[3], className="mb-3", clearable=False),
    
    dbc.Label("2. Upload Data (CSV or Excel)"),
    dcc.Upload(
        id='ra-upload-data',
        children=html.Div(['Drag and Drop or ', html.A('Select Files')]),
        style={'borderWidth': '1px', 'borderStyle': 'dashed', 'borderRadius': '5px', 'textAlign': 'center', 'padding': '10px'},
        className="mb-1"
    ),
    html.Div(id='ra-upload-status', className="small mb-3"),

    dbc.Label("3. Optimization Strategy"),
    dbc.RadioItems(
        id='ra-goal-radio',
        options=[
            {'label': 'Minimize Bill', 'value': 'minimize'},
            {'label': 'Generate Revenue', 'value': 'generate'},
        ],
        value='generate',
        inline=True,
        className="mb-2"
    ),
    html.Div(id='ra-strategy-container', className="mb-3"),

    # This container will be hidden or shown via a callback
    html.Div(id='ra-battery-params-container'),
    
    dbc.Label("Cost Parameters"),
    dbc.InputGroup([dbc.InputGroupText("Supplier €/MWh"), dbc.Input(id='ra-supply-costs', type='number', value=20.0)], className="mb-2"),
    dbc.InputGroup([dbc.InputGroupText("Transport €/MWh"), dbc.Input(id='ra-transport-costs', type='number', value=15.0)], className="mb-3"),
    
    dbc.Button("🚀 Run Analysis", id="ra-run-button", color="primary", n_clicks=0, className="w-100")
]), className="h-100")

# --- Main Content Layout ---
main_content = dbc.Container(fluid=True, children=[
    html.H1("Energy System Simulation ⚡"),
    html.P("Select a system configuration, upload your data, configure the parameters, and run the simulation."),
    html.Hr(),
    html.H4("Selected Configuration"),
    html.Div(id='ra-diagram-output', className="mb-4"),
    html.Hr(),
    # The results will be rendered inside this Div
    html.Div(id="ra-results-output")
])

# --- Full Page Layout with Loading Spinner ---
layout = html.Div([
    dcc.Loading(
        id="ra-fullscreen-loader",
        type="default",
        fullscreen=True,
        # The spinner targets the content of the page
        children=html.Div(id="ra-page-content") 
    ),
    dcc.Store(id='ra-results-store'),
    dcc.Store(id='ra-input-df-store'),
])

# =============================================================================
# Callbacks
# =============================================================================

# This callback renders the main page content. The spinner is outside this, so it's not affected.
@callback(Output('ra-page-content', 'children'), Input('ra-run-button', 'id')) # Trigger on page load
def render_page_content(_):
    return dbc.Container([
        dbc.Row([
            dbc.Col(sidebar, width=12, lg=3, className="mb-4"),
            dbc.Col(main_content, width=12, lg=9),
        ]),
    ], fluid=True, className="mt-4")


@callback(Output('ra-strategy-container', 'children'), Input('ra-goal-radio', 'value'))
def update_strategy_dropdown(goal_choice):
    if goal_choice == 'minimize':
        return html.Div([
            dbc.Alert("Use assets to reduce overall energy costs.", color="info", className="small p-2"),
            dcc.Dropdown(id='ra-strategy-dropdown', options=["Prioritize Self-Consumption", "Optimize on Day-Ahead Market"], value="Prioritize Self-Consumption")
        ])
    else:
        return html.Div([
            dbc.Alert("Actively use assets to trade on energy markets.", color="info", className="small p-2"),
            dcc.Dropdown(id='ra-strategy-dropdown', options=["Simple Battery Trading (Imbalance)", "Advanced Whole-System Trading (Imbalance)"], value="Simple Battery Trading (Imbalance)")
        ])

@callback(Output('ra-battery-params-container', 'children'), Input('ra-situation-dropdown', 'value'))
def show_battery_params(situation):
    display_style = {'display': 'block'}
    if not situation or not ("Battery" in situation or "PAP" in situation):
        display_style = {'display': 'none'}
    return html.Div(style=display_style, children=[
        dbc.Label("Battery Parameters"),
        dbc.InputGroup([dbc.InputGroupText("Power (MW)"), dbc.Input(id='ra-power-mw', type='number', value=1.0, min=0.1, step=0.1)], className="mb-2"),
        dbc.InputGroup([dbc.InputGroupText("Capacity (MWh)"), dbc.Input(id='ra-capacity-mwh', type='number', value=2.0, min=0.1, step=0.1)], className="mb-2"),
        html.Div("Min/Max SoC", className="small text-muted mt-2"),
        dcc.RangeSlider(id='ra-soc-slider', min=0, max=1, step=0.01, value=[0.05, 0.95], marks={0: '0%', 1: '100%'}, tooltip={"placement": "bottom", "always_visible": True}),
        html.Div("Charging Efficiency", className="small text-muted mt-2"),
        dcc.Slider(id='ra-eff-ch', min=0.8, max=1, step=0.01, value=0.95, marks={0.8: '80%', 1: '100%'}, tooltip={"placement": "bottom", "always_visible": True}),
        html.Div("Discharging Efficiency", className="small text-muted mt-2"),
        dcc.Slider(id='ra-eff-dis', min=0.8, max=1, step=0.01, value=0.95, marks={0.8: '80%', 1: '100%'}, tooltip={"placement": "bottom", "always_visible": True}),
        dbc.InputGroup([dbc.InputGroupText("Max Cycles/Year"), dbc.Input(id='ra-max-cycles', type='number', value=600, min=1)], className="mt-2 mb-3"),
    ])

@callback(Output('ra-diagram-output', 'children'), Input('ra-situation-dropdown', 'value'))
def update_diagram(situation):
    if not situation: return dbc.Alert("Please select a system configuration.", color="warning")
    return create_horizontal_diagram_with_icons(situation)

@callback(
    [Output('ra-upload-status', 'children'), Output('ra-input-df-store', 'data')],
    Input('ra-upload-data', 'contents'), State('ra-upload-data', 'filename'),
    prevent_initial_call=True
)
def handle_upload(contents, filename):
    if contents is None: return no_update, no_update
    df, message = parse_contents(contents, f"revenue_{filename}")
    if df is None: return html.Div(message, className="text-danger small"), None
    return html.Div(message, className="text-success small"), df.to_json(date_format='iso', orient='split')

@callback(
    Output('ra-results-store', 'data'),
    Input('ra-run-button', 'n_clicks'),
    [
        State('ra-input-df-store', 'data'), State('ra-strategy-dropdown', 'value'),
        State('ra-power-mw', 'value'), State('ra-capacity-mwh', 'value'),
        State('ra-soc-slider', 'value'), State('ra-eff-ch', 'value'),
        State('ra-eff-dis', 'value'), State('ra-max-cycles', 'value'),
        State('ra-supply-costs', 'value'), State('ra-transport-costs', 'value'),
    ],
    prevent_initial_call=True
)
def run_model(n_clicks, df_json, strategy, power_mw, cap_mwh, soc, eff_ch, eff_dis, cycles, supply, transport):
    if not df_json: return {"error": "Please upload a data file first."}
    input_df = pd.read_json(df_json, orient='split')
    params = {
        "POWER_MW": power_mw or 0, "CAPACITY_MWH": cap_mwh or 0,
        "MIN_SOC": soc[0] if soc else 0, "MAX_SOC": soc[1] if soc else 1,
        "EFF_CH": eff_ch or 1, "EFF_DIS": eff_dis or 1,
        "MAX_CYCLES": cycles or 0, "INIT_SOC": 0.5,
        "SUPPLY_COSTS": supply or 0, "TRANSPORT_COSTS": transport or 0,
        "STRATEGY_CHOICE": strategy, "TIME_STEP_H": 0.25
    }
    return run_revenue_model_placeholder(params, input_df, lambda msg: print(msg))

@callback(Output('ra-results-output', 'children'), Input('ra-results-store', 'data'))
def display_results(results_data):
    if not results_data:
        return dbc.Alert("Configure your simulation in the sidebar and click 'Run Analysis' to see the results.", color="info")
    if results_data.get("error"):
        return dbc.Alert(f"Error: {results_data['error']}", color="danger")
    
    summary = results_data["summary"]
    df = pd.read_json(results_data["df"], orient='split')
    total_result_col = find_total_result_column(df)
    net_result = df[total_result_col].sum() if total_result_col and not df.empty else 0
    warnings_layout = [dbc.Alert(w, color="warning") for w in results_data.get("warnings", [])]
    return html.Div([
        dcc.Download(id="ra-download-excel"),
        html.H4("📈 Results Summary"),
        dbc.Alert(f"Analysis Method Used: {summary.get('optimization_method', 'N/A')}", color="secondary"),
        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([html.H6("Net Result / Revenue", className="text-muted small"), html.H4(f"€ {net_result:,.0f}")]))),
            dbc.Col(dbc.Card(dbc.CardBody([html.H6("Total Cycles", className="text-muted small"), html.H4(f"{summary.get('total_cycles', 0):.1f}")]))),
            dbc.Col(dbc.Card(dbc.CardBody([html.H6("Infeasible Days", className="text-muted small"), html.H4(f"{len(summary.get('infeasible_days', []))}")]))),
        ], className="mb-3"),
        *warnings_layout,
        dbc.Button("📥 Download Full Results (Excel)", id="ra-download-button", color="success", className="mt-3 mb-4"),
        html.Hr(),
        html.H4("📊 Interactive Charts"),
        dcc.Dropdown(id='ra-resolution-dropdown', options=['15 Min (Original)', 'Hourly', 'Daily', 'Monthly', 'Yearly'], value='Daily', clearable=False, className="mb-3"),
        dbc.Tabs(id="ra-charts-tabs")
    ])

@callback(
    Output('ra-charts-tabs', 'children'),
    [Input('ra-resolution-dropdown', 'value'), Input('ra-results-store', 'data')]
)
def update_charts(resolution, results_data):
    if not results_data or results_data.get("error"):
        return [dbc.Tab(label="No Data", children=[dbc.Alert("Run a successful analysis to view charts.", color="info")])]
    df_original = pd.read_json(results_data["df"], orient='split')
    df_original.index = pd.to_datetime(df_original.index)
    if df_original.empty:
        return [dbc.Tab(label="No Data", children=[dbc.Alert("Model returned no data for plotting.", color="warning")])]

    df_resampled = resample_data(df_original.copy(), resolution)
    total_result_col = find_total_result_column(df_original)
    tab1_content = dcc.Graph(figure=px.line(df_resampled, y=total_result_col, title=f"Financial Result ({resolution})")) if total_result_col else dbc.Alert("No 'total_result' column found.", color="warning")
    
    tab2_children = []
    if 'production_PV' in df_resampled.columns: tab2_children.append(dcc.Graph(figure=px.line(df_resampled, y='production_PV', title=f"PV Production ({resolution})")))
    if 'load' in df_resampled.columns: tab2_children.append(dcc.Graph(figure=px.line(df_resampled, y='load', title=f"Load ({resolution})")))
    tab3_content = dcc.Graph(figure=px.line(df_original, y='SoC_kWh', title="Battery SoC (15 Min Resolution)")) if 'SoC_kWh' in df_original.columns else dbc.Alert("No 'SoC_kWh' column found.", color="warning")

    return [
        dbc.Tab(tab1_content, label="💰 Financial Results"),
        dbc.Tab(tab2_children, label="⚡ Energy Profiles"),
        dbc.Tab(tab3_content, label="🔋 Battery SoC"),
    ]

@callback(
    Output("ra-download-excel", "data"),
    Input("ra-download-button", "n_clicks"),
    State("ra-results-store", "data"),
    prevent_initial_call=True,
)
def download_excel(n_clicks, results_data):
    if not results_data or "output_file_bytes" not in results_data: return no_update
    file_bytes = results_data["output_file_bytes"]
    return dcc.send_bytes(file_bytes, f"Revenue_Analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx")