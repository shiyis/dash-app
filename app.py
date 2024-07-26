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


app = dash.Dash(__name__, use_pages=True, external_stylesheets=[dbc.themes.SPACELAB], suppress_callback_exceptions=True)
server = app.server

# Define the menubar
menubar = dbc.NavbarSimple(
    children=[
        # dbc.NavItem(dbc.NavLink("Election EDA", href="/", id="election-eda")),
        dbc.NavItem(dbc.NavLink("Measuring Ideal Points", href="page2", id="measuring-ideal-points")),
        dbc.NavItem(dbc.NavLink("Supplementary Blog", href="https://shiyis.github.io/politics-docs", id="supplementary-blog", external_link=True)),
        dbc.NavItem(dbc.NavLink("GitHub Repo", href="https://github.com/shiyis/politics", external_link=True)),

    ],
    brand="POLITICS: A Political Opinions, Language, And Ideology Text Interpretation And Classification Solution",
    brand_href="/",
    color="light",
    dark=True,
    fluid=True,
    style= {

        "position": "relative",
        "padding": "0.5rem 4.2rem 0.5rem 4.2rem",
        "whiteSpace": "pre-wrap",
        "font-family": "system-ui",
        "text-shadow": "0px 1px 0 #ccc",
        "color": "dimgray"

    }
)

content = html.Div([dash.page_container], id="page-content")
# Define the app layout
app.layout = html.Div([
    menubar,
    content
])

if __name__ == "__main__":
    app.run_server(debug=True, port=8050)
