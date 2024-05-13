import dash
from dash import html
import dash
import pandas as pd
import os
import dash_bootstrap_components as dbc
from dash import dcc, html, dash_table, Input, Output, State, callback
import dash_leaflet as dl
from datetime import datetime

dash.register_page(
    __name__, path="/", title="Project Intro & Roadmap", location="sidebar"
)



PAGE_STYLE = {
    "position": "relative",
    "margin": "-5.5rem 4rem 0rem 20rem",
    "color": "#000",
    "text-shadow": "#000 0 0",
    "whiteSpace": "pre-wrap",
    "font-family": "system-ui",
}

IFRAME_STYLE = {

    "margin": "0rem -5rem 0rem -5rem",

}


layout = html.Div(
    [
#         html.H5("This Is A Political Text Ideology Extraction Tool (or Namely POLITIX)"),
#         html.Hr(),
#         html.P([
#             """This tool tries to help someone who is interested in voting activities understand the political leaning of a candidate for federal elections through quantitative measures and data-driven procedures. """
#         ,"""It draws inspiration from websites like """, html.A("Opensecrets", href="http://opensecrets.org"), """, """, html.A("Voteview", href="http://voteview.com"),""" and many other relevant sources (listed in the repo), where it strives to uncover information of a politician's stance, agenda, and activities (campaign-related or financial)."""]
#         ),
#         html.P([
#             """It tries to help the general population who is interested in partaking in political activities understand a politician's (or anyone who authors political content) stance by extracting crucial information from relevant political text. All src code and relevant doc is hosted on """, html.A("here", href="https://github.com/shiyis/politix"), """. This project seeks to conduct the research with full transparency and abide to relevant code of conduct."""]
#         ),
#         html.Br(),

#         html.H5("The Topic This Project Tries to Explore"),
#         html.Hr(),
#         html.P([
#             """Measuring political sentiment and polarization is a common practice in the realm of social science research. However, it may also be applicable to solving business problems, like providing more information about a certain candidate to voters to fill the information gap and facilitate voting processes."""]
#         ),
#         dcc.Markdown([
#             """Websites provided above offer valuable educational information to start. This project tries to top it off by retrieving organic information (tweets) about said candidates and conducting analysis accordingly. Although it seems to be framed as an open ended project. There's still some downstream workflow that will be demonstrated and clarified.\n\n❓ The One Million Dollar Question: How Is Political Subjectivity Measured? \n\n _There are at a very surface level three types of political polarizations: Elite, Mass, and Affective_. \n\nThis project tries to understand the "Elite" type of polarization which is by definition: polarization between the party-in-government, party-in-opposition, legislators, and/or lawmakers. Content below will demonstrate how we approach this,
#             """,
      
# ]
#         ),      
        html.Div(
                [
                    html.Iframe(
                        src="https://copy-of-politix-a-politi-mzz410r.gamma.site/",
                        style={
                            "height": "2500px",
                            "width": "100%",
                            "font-family": "system-ui",
                            "padding": "0 0 0 0",
                            "--card-width":"90rem",
                            "z-index": "1",
                        },
                    ),
                ],
                className="home",
                style=IFRAME_STYLE,
            )
    ],
    className="page1",
    id="page1-content",
    style=PAGE_STYLE,
)




