import dash
import pandas as pd
import os
import dash_bootstrap_components as dbc
from dash import dcc, html, Output, callback
import pandas as pd
import numpy as np
import os
import dash_ag_grid as dag
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import warnings

# Suppress regex match group warning
warnings.filterwarnings(
    "ignore",
    message="This pattern is interpreted as a regular expression, and has match groups.",
)

# Your code here


plt.switch_backend("Agg")
states = pd.read_csv("./data/states.csv")
candidates = pd.read_csv("./data/2022/processed_weball.csv")

dash.register_page(__name__, title="Project POLITICS | Measuring Subjectivity", location="sidebar")


PAGE_STYLE = {
    "position": "absolute",
    "margin": "2.5rem 5rem 0rem 5rem",
    "color": "#000",
    # "text-shadow": "#000 0 0",
    "whiteSpace": "pre-wrap",
}

custom = "./data/custom_data/clean/author_map.txt"


if not os.path.isfile(custom):
    dataPath = "./data/2022/candidate-tweets-2020/clean/"
else:
    dataPath = "/".join(custom.split("/")[:-1])


authors = pd.read_csv("./data/2022/authors.csv")
authors["name"] = authors["name"].str.replace("\n", "")


layout = html.Div(
    [
        html.H5("Measuring Political Stance and Subjectivity with VAE methods"),
        html.Hr(),
        html.P(
            [
                dcc.Markdown(
                    """To continue with our project objective of measuring and investigating how a politician's internal motivation aligns with their external actions (fundraising, disbursements, and various expenditures with respect to their political acitivities and agenda; we have disclosed the most basic information in the previous section, or [Exploratory Data Analysis of General Election Candidacy](https://my-dash-app-ilf47zak6q-uc.a.run.app), this section will present a statistical topic modeling over the authors/politicians' tweets."""
                ),
                html.A(
                    "TBIP or Text-based Ideal Point Model",
                    href="https://www.aclweb.org/anthology/2020.acl-main.475/",
                ),
                """ is an unsupervised probabilistic topic model (Keyon V., Suresh N., David B. et al.) that evaluates texts to quantify the political stances of their authors. The model does not require any text labeled with an ideology, nor does it use political parties or votes.""",
                html.P(""""""),
                dcc.Markdown(
                    """Instead, it assesses the `latent political viewpoints` of text writers and how `per-topic word choice` varies according to the author's political stance `("ideological topics")` given a corpus of political text and the author of each document."""
                ),
                """Below are the resulting ideal points,\n""",
                dbc.Row(
                    [
                        dbc.Col(dbc.Row(children=[],id="bar-graph-plotly")),
                        dbc.Col(
                            children=[
                                dag.AgGrid(
                                    id="grid",
                                    rowData=authors.to_dict("records"),
                                    columnDefs=[{"field": i} for i in authors.columns],
                                    columnSize="sizeToFit",
                                    style={"text-align": "center"},
                                )
                            ],
                            md=20,
                            style={"margin": "2rem 0 0 0"},
                        ),
                    ],
                    className="mt-4",
                    style={"text-align": "center"},
                ),
        ]),
        html.Br(),
    ],
    className="page2",
    style=PAGE_STYLE,
)


@callback(
    Output("cand-names-row-2", "children"),
    [dash.dependencies.Input("state-dropdown", "value")],
)
def update_output(value):
    a = states.loc[states["name"] == value, "state"]
    if value:
        res = candidates.loc[
            candidates["State"] == a.iloc[0], "Candidate name"
        ].tolist()
    else:
        res = candidates["Candidate name"].tolist()
    return html.Label(
        ["Select Candidate"],
        style={
            "font-size": "13px",
            "text-align": "left",
            "off-set": 4,
            "color": "#808080",
        },
    ), dcc.Dropdown(res, id="names-dropdown", searchable=True, multi=True)


@callback(
    Output("bar-graph-plotly", "children"),
    [dash.dependencies.Input("bar-graph-plotly", "figure")],
)
def my_callback(figure_empty):
    fig = px.scatter(
        authors, x=["ideal_point"], y=[1] * len(authors), hover_data=["name"]
    )
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)

    string_1 = "(D)"
    string_2 = "(R)"
    string_3 = "(I)"

    x1 = authors[authors["name"].str.contains(rf"\b{string_1}\b")]
    x2 = authors[authors["name"].str.contains(rf"\b{string_2}\b")]
    x3 = authors[authors["name"].str.contains(rf"\b{string_3}\b")]

    mr = x2[x2["ideal_point"] < 0]
    mb = x1[x1["ideal_point"] < 0]
    pr = x2[x2["ideal_point"] >= 0]
    pb = x1[x1["ideal_point"] >= 0]

    mi = x3[x3["ideal_point"] < 0]
    pi = x3[x3["ideal_point"] >= 0]

    y = np.array([1] * len(authors))

    layout = go.Layout(
        xaxis={
            "title": "Author's Ideal Point from Moderate to Progressive",
            "visible": True,
            "showticklabels": True,
        },
        yaxis={"title": "y-label", "visible": False, "showticklabels": False},
        height=230,
        margin=dict(l=0, r=0, t=0, b=0),
        plot_bgcolor="#f2f2f2",  # Set the background color of the plot
        # paper_bgcolor='#f2f2f2'  # Set the background color of the entire plot area
    )

    fig = go.Figure(layout=layout)

    fig.add_trace(
        go.Scatter(
            x=np.array(mr["ideal_point"]),
            y=y,
            mode="markers",
            name="moderate repub",
            marker=dict(symbol="x", color="red"),
            hovertemplate="<br><b>Ideal Point</b>: %{x}<br>" + "<b>%{text}</b>",
            text=["Author: {}".format(i) for i in mr["name"].tolist()],
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=np.array(mb["ideal_point"]),
            y=y,
            mode="markers",
            name="conservative",
            marker=dict(color="blue"),
            hovertemplate="<br><b>Ideal Point</b>: %{x}<br>" + "<b>%{text}</b>",
            text=["Author: {}".format(i) for i in mb["name"].tolist()],
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=np.array(mi["ideal_point"]),
            y=y,
            mode="markers",
            name="conservative",
            marker=dict(symbol="square", color="grey"),
            hovertemplate="<br><b>Ideal Point</b>: %{x}<br>" + "<b>%{text}</b>",
            text=["Author: {}".format(i) for i in mi["name"].tolist()],
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=np.array(pi["ideal_point"]),
            y=y,
            mode="markers",
            name="conservative",
            marker=dict(symbol="square", color="grey"),
            hovertemplate="<br><b>Ideal Point</b>: %{x}<br>" + "<b>%{text}</b>",
            text=["Author: {}".format(i) for i in pi["name"].tolist()],
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=np.array(pr["ideal_point"]),
            y=y,
            mode="markers",
            name="conservative",
            marker=dict(symbol="x", color="red"),
            hovertemplate="<br><b>Ideal Point</b>: %{x}<br>" + "<b>%{text}</b>",
            text=["Author: {}".format(i) for i in pr["name"].tolist()],
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=np.array(pb["ideal_point"]),
            y=y,
            mode="markers",
            name="conservative",
            marker=dict(color="blue"),
            hovertemplate="<br><b>Ideal Point</b>: %{x}<br>" + "<b>%{text}</b>",
            text=["Author: {}".format(i) for i in pb["name"].tolist()],
            showlegend=False,
        )
    )

    return [dcc.Graph(id="bar-graph-plotly", figure=fig)]
