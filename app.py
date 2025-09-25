# app.py
import dash
from dash import dcc, html
import dash_bootstrap_components as dbc

# --- App Instantiation ---
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME],
    suppress_callback_exceptions=True,
    use_pages=True,  # Enables the pages/ folder functionality
)
server = app.server

# --- Navigation Bar ---
navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("Home", href="/")),
        dbc.DropdownMenu(
            children=[
                dbc.DropdownMenuItem("Business Case Modeler", href="/business-case"),
                dbc.DropdownMenuItem("Revenue Analysis", href="/revenue-analysis"),
                dbc.DropdownMenuItem("Battery Sizing Tool", href="/battery-sizing"),
            ],
            nav=True,
            in_navbar=True,
            label="Tools",
        ),
    ],
    brand="Flink EMS",
    brand_href="/",
    color="primary",
    dark=True,
    className="mb-4",
)

# --- Main App Layout ---
app.layout = html.Div([
    navbar,
    # This component holds the content of the different pages
    dash.page_container
])

# --- Run the App ---
if __name__ == '__main__':
    app.run(debug=True)