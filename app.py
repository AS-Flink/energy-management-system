# app.py
import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc

# --- App Instantiation ---
# Using a Bootstrap theme for a clean layout and responsive design.
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True, # Important for multi-page apps
    use_pages=True, # Enables the pages/ folder functionality
)
server = app.server

# --- Main App Layout ---
app.layout = dbc.Container([
    # This component holds the content of the different pages
    dash.page_container
], fluid=True)


# --- Run the App ---
if __name__ == '__main__':
    app.run(debug=True)