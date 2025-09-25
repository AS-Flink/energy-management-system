# app.py (Corrected Version)
import dash
from dash import dcc, html
import dash_bootstrap_components as dbc

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP, '/assets/styles.css'],
    suppress_callback_exceptions=True,
    use_pages=True,
    assets_folder='assets'
)
server = app.server

navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("Home", href="/")),
        dbc.DropdownMenu(
            children=[
                dbc.DropdownMenuItem("Business Case Modeler", href="/business-case"),
                dbc.DropdownMenuItem("Revenue Analysis", href="/revenue-analysis"),
                dbc.DropdownMenuItem("Battery Sizing Tool", href="/battery-sizing"),
            ],
            nav=True, in_navbar=True, label="Tools",
        ),
    ],
    brand="Flink EMS", brand_href="/", color="primary", dark=True, className="mb-4",
)

app.layout = html.Div([
    navbar,
    dash.page_container
])

if __name__ == '__main__':
    app.run(debug=True)