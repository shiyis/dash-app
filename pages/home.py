import dash
from dash import html
import dash
import pandas as pd
import os
import dash_bootstrap_components as dbc
from dash import dcc, html, dash_table, Input, Output, State, callback
import dash_leaflet as dl
from datetime import datetime

# dash.register_page(
#     __name__, path="/", title="Project Intro & Roadmap", location="sidebar"
# )
PAGE_STYLE = {
    "background-color": "#fff",
    "position": "relative",
    "margin": "-7.5rem -5rem 0rem 8rem",
    "width": "100%",
    "color": "#000",
    "fontfamily": "system-ui",
}
layout = html.Div(
    [
        html.Iframe(
            src="https://gamma.app/embed/yjmv7s7hjm5zyau",
            style={
                "height": "2500px",
                "width": "100%",
                "font-family": "system-ui",
                "z-index": "1",
            },
        ),
    ],
    className="home",
    style=PAGE_STYLE,
)
