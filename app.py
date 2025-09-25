# app.py (Simplified Version)
import dash
from dash import dcc, html
import dash_bootstrap_components as dbc

# Simple app instantiation without any managers
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True,
    use_pages=True,
    assets_folder='assets'
)
server = app.server

# Navigation Bar (no changes)
navbar = dbc.NavbarSimple(
    brand="Flink EMS", brand_href="/", color="primary", dark=True, className="mb-4",
    children=[
        dbc.NavItem(dbc.NavLink("Home", href="/")),
        dbc.DropdownMenu(
            label="Tools", nav=True, in_navbar=True,
            children=[
                dbc.DropdownMenuItem("Business Case Modeler", href="/business-case"),
                dbc.DropdownMenuItem("Revenue Analysis", href="/revenue-analysis"),
                dbc.DropdownMenuItem("Battery Sizing Tool", href="/battery-sizing"),
            ],
        ),
    ]
)

# Main App Layout (no changes)
app.layout = html.Div([
    navbar,
    dash.page_container
])

# Run the App
if __name__ == '__main__':
    app.run(debug=True)