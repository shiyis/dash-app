import dash
from dash import html
import dash
import pandas as pd
import os
import dash_bootstrap_components as dbc
from dash import dcc, html, dash_table, Input, Output, State, callback
import dash_leaflet as dl
from datetime import datetime
# 
# dash.register_page(
#     __name__, path="/", title="Project Intro & Roadmap", location="sidebar"
# )
#

PAGE_STYLE = {
    # "margin": "0rem 5rem 0rem 5rem",
    "font-family": "system-ui",
    "background-color": "#999",
    # "padding-top": "3rem",
}


layout = html.Div(
    [
        html.Div(
            [
                html.Iframe(
                    src="https://gamma.app/embed/l9741p0k5hh1zlo",
                    style={
                        "width": "1600px",
                        "max-with" : "100%",
                        "height": "2300px",
                        "display": "block",
                        "margin": "0 auto",
                        "border-radius": "5px",
                    },
                )
            ],
        )
    ],
    className="home",
    id="home-content",
    style=PAGE_STYLE,
)
