import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
import pandas as pd
import dash_leaflet.express as dlx
from dash import dcc, html, Output, callback, callback_context
import dash_leaflet as dl
from dash_extensions.javascript import arrow_function
from dash_extensions.javascript import assign
from dash import dash_table


app = dash.Dash(__name__, use_pages=True, external_stylesheets=[dbc.themes.SPACELAB])
server = app.server

# Define the menubar
menubar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("Election EDA", href="/", id="election-eda")),
        dbc.NavItem(dbc.NavLink("Ideal Points", href="page2", id="measuring-ideal-points")),
        dbc.NavItem(dbc.NavLink("GitHub Repo", href="https://github.com/shiyis/politix", external_link=True)),

    ],
    brand="POLITIX: A Political Text Ideology Extraction Project",
    brand_href="/",
    color="primary",
    dark=True,
    fluid=True,
    style= {

        "position": "relative",
        # "margin": "0rem 5rem 0rem 5rem",
        "padding": "0.5rem 4.2rem 0.5rem 4.2rem",
        "color": "#000",
        "text-shadow": "#000 0 0",
        "whiteSpace": "pre-wrap",
        "font-family": "system-ui",

    }
)

content = html.Div([dash.page_container], id="page-content")
# Define the app layout
app.layout = html.Div([
    menubar,
    content
])

if __name__ == "__main__":
    app.run_server(debug=False, port=8050)
