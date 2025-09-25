import dash
from dash import html, dcc
import dash_bootstrap_components as dbc

# This registers the page with Dash and sets its URL path to "/" and its name in the navigation.
dash.register_page(__name__, path='/', name='Home 🏠')

# This is the layout for the home page.
layout = dbc.Container([
    # Page Title
    dbc.Row(
        dbc.Col(html.H1("Flink Energy Management System (EMS) Simulation", className="text-center my-4"), width=12)
    ),
    html.Hr(),

    # Main content area with two columns
    dbc.Row([
        # Left Column: Navigation Cards
        dbc.Col([
            html.H3("Tools"),
            html.P("Please select a tool from the navigation bar or the links below to begin."),
            html.Br(),
            
            # Card for the Business Case Modeler
            dbc.Card([
                dbc.CardHeader(html.H4("📈 Financial Modeling")),
                dbc.CardBody([
                    html.P("Manage and run detailed business case simulations for BESS and PV projects."),
                    dcc.Link(dbc.Button("Go to Business Case Modeler", color="primary"), href="/business-case"),
                ]),
            ], className="mb-4"),

            # Card for the Revenue Analysis Tool
             dbc.Card([
                dbc.CardHeader(html.H4("💰 Revenue Analysis")),
                dbc.CardBody([
                    html.P("Simulate potential revenue streams based on different system configurations and market strategies."),
                    dcc.Link(dbc.Button("Go to Revenue Analysis", color="primary"), href="/revenue-analysis"),
                ]),
            ], className="mb-4"),

            # Card for the Battery Sizing Tool
            dbc.Card([
                dbc.CardHeader(html.H4("🛠️ Sizing Tools")),
                dbc.CardBody([
                    html.P("Calculate the optimal battery power and capacity for peak shaving applications."),
                    dcc.Link(dbc.Button("Go to Battery Sizing Tool", color="primary"), href="/battery-sizing"),
                ]),
            ], className="mb-4"),
        ], width=12, md=5), # Takes full width on mobile, 5/12 on medium screens and up

        # Right Column: Image
        dbc.Col(
            html.Img(src="https://i.postimg.cc/2ykmvjVb/Energy-blog-anim.gif", className="img-fluid rounded"),
            width=12, md=7, className="mt-4 mt-md-0"
        )
    ], className="align-items-center") # Vertically aligns the columns
], fluid=True, className="mt-5")