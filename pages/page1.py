import dash
import pandas as pd
import dash_bootstrap_components as dbc
import dash_leaflet.express as dlx
from dash import dcc, html, Output, callback, callback_context
import dash_leaflet as dl
from dash_extensions.javascript import arrow_function
from dash_extensions.javascript import assign
from dash import dash_table
import dash_leaflet as dl
import dash_extensions.javascript as dj


dash.register_page(
    __name__, path="/", title="POLITICS | Exploratory Data Analysis", location="sidebar"
)
pd.set_option("float_format", "{:.2f}".format)


# input data
candidates = pd.read_csv("./data/2022/processed_weball.csv")
states = pd.read_csv("./data/states.csv")

# define page style
PAGE_STYLE = {
    "position": "relative",
    "margin": "2.5rem 5rem 0rem 5rem",
    "color": "#000",
}


def get_info(feature=None):
    header = [
        html.H4(
            "PAC Funds Spent and Raised by States\n",
            style={
                "color": "#01579b",
                "font-family": "system-ui",
                "font-weight": "bold",
                "font-size": "12px",
            },
        )
    ]
    if not feature:
        return header + [
            html.B(
                "Hover over a state",
                style={"font-family": "system-ui", "font-size": "12px"},
            )
        ]
    return [
        html.B(feature["properties"]["name"]),
        html.Br(),
        html.Div(
            [
                "Total Received: ",
                html.Span(
                    str("${:.3f}".format(feature["properties"]["total_r"]) + "\n({}M+)".format(int(feature["properties"]["total_r"] / 1e6)) if feature["properties"]["total_r"] != 0 else "0"),
                    style={
                        "font-size": "14px",
                        "font-weight": "bold",
                        "display": "inline-block",
                        "width": "100%",
                    },
                ),
            ]
        ),
        html.Br(),
        html.Div(
            [
                "Total Spent: ",
                html.Span(
                    str("${:.3f}".format(feature["properties"]["total_s"]) + "\n({}M+)".format(int(feature["properties"]["total_r"] / 1e6)) if feature["properties"]["total_r"] != 0 else "0"),
                    style={
                        "font-size": "14px",
                        "font-weight": "bold",
                        "display": "inline-block",
                        "width": "100%",
                    },
                ),
            ]
        ),
    ]

def create_choropleth(id="geojson1", info_id="info1"):
    classes = [0, 5000000, 10000000, 50000000, 100000000, 200000000]

    # New purple color scale
    # Green color scale
    colorscale = [
        "#fff",
        "#e0f7ff",  # Very light blueberry
        "#b3e5fc",  # Light blueberry
        "#4fc3f7",  # Medium blueberry
        "#0288d1",  # Deep blueberry
        "#01579b",  # Dark blueberry
        "#003f7f"   # Very dark blueberry
    ]





    style = dict(weight=1, opacity=1, color="white", dashArray="", fillOpacity=0.6)

    # Create colorbar with the new purple color scale.
    ctg = [
        "${}M+".format(int(cls / 1e6)) if cls != 0 else "0"
        for cls in classes[:-1]
    ] + ["${}M+".format(int(classes[-1] / 1e6))]

    colorbar = dlx.categorical_colorbar(
        categories=ctg,
        colorscale=colorscale,
        width=520,
        height=10,
        position="bottomleft",
        style={"fill-opacity": "0.4"},
    )

    # Geojson rendering logic
    style_handle = assign(
        """function(feature, context){
        const {classes, colorscale, style, colorProp} = context.hideout;
        const value = feature.properties[colorProp];
        for (let i = 0; i < classes.length; ++i) {
            if (value > classes[i]) {
                style.fillColor = colorscale[i];
            }
        }
        return style;
        }"""
    )

    # Create geojson.
    geojson = dl.GeoJSON(
        url="/assets/us-states.json",
        style=style_handle,
        zoomToBounds=False,
        zoomToBoundsOnClick=False,
        hoverStyle=arrow_function(
            dict(weight=3, color="purple", opacity=0.5, fillOpacity=1, dashArray="")
        ),
        hideout=dict(
            colorscale=colorscale, classes=classes, style=style, colorProp="total_r"
        ),
        id=id,
    )

    # Create info control.
    info = html.Div(
        children=get_info(),
        id=info_id,
        className="info",
        style={
            "position": "absolute",
            "top": "300px",
            "left": "650px",
            "width": "129px",
        },
    )
    return geojson, colorbar, info


# creating chloropleth
choropleth1 = create_choropleth()
choropleth2 = create_choropleth(id="geojson2", info_id="info2")

style_url = "https://api.mapbox.com/styles/v1/shiyis/cm1dvrpkt000201nt42nkgugq/tiles/{z}/{x}/{y}?access_token=pk.eyJ1Ijoic2hpeWlzIiwiYSI6ImNtMWMxdXN0YTB0djUybG9tN2Rqbmlyd3gifQ.b0XJVI-QJJvEeJWzO1Gdgw"
attribution = "&copy; <a href='https://www.mapbox.com/'>Mapbox</a>"

map1 = dl.Map(
    children=[dl.TileLayer(url=style_url,
                     attribution=attribution)],
    style={"height": "450px", "margin-top": "0rem", "background-color": "#fff"},
    center=[39, -98],
    zoom=4,
    id="candidates-stats-marker",
)
map2 = dl.Map(
    children=[dl.TileLayer(url=style_url,
                     attribution=attribution)],
    style={"height": "450px", "margin-top": "0rem", "background-color":"#fff"},
    center=[
        states[states["state"] == "DC"]["latitude"].iloc[0],
        states[states["state"] == "DC"]["longitude"].iloc[0],
    ],
    zoom=7,
    id="candidates-individual-marker",
)

def format_currency(value):
    if value >= 1_000_000_000:
        return f"${value / 1_000_000_000:.2f}B"
    elif value >= 1_000_000:
        return f"${value / 1_000_000:.2f}M"
    else:
        return f"${value:,.2f}"


table1 = (
    candidates[["Party affiliation", "Total receipts"]]
    .groupby("Party affiliation")
    .agg("sum")
    .sort_values("Total receipts")[::-1][:5]
    .reset_index()
    .rename(
        columns={
            "Candidate state": "State",
            "Total receipts": "Total Receipts",
        }
    )
    .round(2)

)

table1["Total Receipts"] = table1["Total Receipts"].apply(format_currency)

table2 = (
    candidates[["Candidate state", "Party affiliation", "Affiliated Committee Name"]]
    .groupby(["Candidate state", "Party affiliation"])
    .agg("count")
    .sort_values("Affiliated Committee Name")
    .fillna("None")[::-1][:5]
    .reset_index()
    .rename(
        columns={
            "Affiliated Committee Name": "# Affiliated Cmt.",
        }
    )
    .round(2)
)
table3 = (
    candidates[["Candidate state", "Party affiliation", "Total receipts"]]
    .groupby(["Candidate state", "Party affiliation"])
    .agg("sum")
    .sort_values("Total receipts")[::-1][:5]
    .reset_index()
    .rename(
        columns={
            "Candidate state": "State",
            "Total receipts": "Total Receipts",
        }
    )
    .round(2)
)

table3["Total Receipts"] = table3["Total Receipts"].apply(format_currency)

checkbox_options = [
    {"label": " Avg Raised", "value": "Avg Raised"},
    {"label": " Avg Spent", "value": "Avg Spent"},
    {"label": " Total Raised", "value": "Total Raised"},
    {"label": " Total Spent", "value": "Total Spent"},
    {"label": " N_PACs", "value": "# PACs"},
]


layout = html.Div(
    [
        html.H5("Exploratory Data Analysis of General Election Candidacy"),
        html.Hr(),
        html.P(
            """This part of the project will first present the receipts, disbursements, and other expenditures of a politician in visualization format grounded in states; for example, how many different political action committees there are by US states. This part of the project will also break down all the candidates of 2022 their basic information including their basic demographics, political party affiliation, election cycle, and incumbency."""
        ),
        dcc.Markdown(
            """All info is retrievable through the Federal Election Commission's directory. Please check out this [link](https://shiyis.github.io/politics-docs/) for full documentation.
            Also, please check out this [link]() for some qualitative analyses answering some questions with respect to the visualizations created below.""",
        ),
        dbc.Row(
            [
                dbc.Col(
                    dbc.Row(
                        children=[
                            html.Label(
                                ["Select State"],
                                style={
                                    "font-size": "13px",
                                    "text-align": "left",
                                    "off-set": 4,
                                    "color": "#000",
                                },
                            ),
                            dcc.Dropdown(
                                pd.DataFrame(pd.read_csv("./data/states.csv"))[
                                    "name"
                                ].tolist(),
                                id="state-dropdown",
                            ),
                        ],
                        id="states-row",
                    )
                ),
                dbc.Col(
                    dbc.Row(
                        children=[
                            html.Label(
                                ["Select Candidate"],
                                style={
                                    "font-size": "13px",
                                    "text-align": "left",
                                    "off-set": 4,
                                    "color": "#000",
                                },
                            ),
                            dcc.Dropdown(id="names-dropdown"),
                        ],
                        id="cand-names-row",
                    )
                ),
            ]
        ),
        html.Br(),
        dbc.Row(
            [
                dbc.Col(dbc.Row(dbc.Col(children=[map1])), id="map1-col"),
                dbc.Col(dbc.Row(dbc.Col(children=[map2])), id="map2-col"),
            ]
        ),
        dbc.Row(
            [
                dbc.Col(
                    dbc.Row(
                        children=[
                            html.Div(
                                children=[
                                    html.Label(
                                        [
                                            "PAC Funds Raised and Spent between 2021-2022"
                                        ],
                                        style={
                                            "font-size": "13px",
                                            "text-align": "left",
                                            "off-set": 4,
                                            "color": "#000",
                                            "margin": "2.6rem 0rem 0rem 0rem",
                                        },
                                    ),
                                    html.Div(
                                        dcc.Slider(
                                            7000,
                                            27500000,
                                            2500000,
                                            value=0,
                                            id="pac-exp-filter",
                                        ),
                                        style={"margin": "0.5rem -1.3rem 0rem -1.3rem"},
                                    ),
                                ],
                                id="slider-1",
                            )
                        ],
                        id="states-row",
                    )
                ),
                dbc.Col(
                    dbc.Row(
                        dbc.Col(
                            children=[
                                html.Div(
                                    [
                                        dcc.Checklist(
                                            id="checkboxes",
                                            options=checkbox_options,
                                            value=[
                                                i["value"] for i in checkbox_options
                                            ],
                                            inline=True,
                                            labelStyle={
                                                "display": "inline-block",
                                                "width": "20%",
                                                "font-size": "13px",
                                                "text-align": "left",
                                                "color": "#000",
                                                "text-align": "right",
                                                "backgroundColor": "rgb(207, 216, 220)",
                                            },
                                        ),
                                        html.Table(
                                            [
                                                html.Tr(
                                                    [
                                                        html.Td(
                                                            dcc.Checklist(
                                                                id="parties-checkbox",
                                                                options=[
                                                                    {
                                                                        "label": " REP",
                                                                        "value": "REP",
                                                                    },
                                                                    {
                                                                        "label": " DEM",
                                                                        "value": "DEM",
                                                                    },
                                                                    {
                                                                        "label": " 3RD",
                                                                        "value": "3RD",
                                                                    },
                                                                ],
                                                                value=[
                                                                    "REP",
                                                                    "DEM",
                                                                    "3RD",
                                                                ],
                                                            ),
                                                            style={
                                                                "vertical-align": "middle",
                                                                "width": "10.5%",
                                                                "padding": "5px",
                                                            },
                                                        ),
                                                        html.Tbody(
                                                            [
                                                                html.Tr(
                                                                    id="table-row-1",
                                                                    style={
                                                                        "border-left": "1px solid #ddd",
                                                                        "border-top": "1px solid #ddd"
                                                                    },
                                                                )
                                                            ]
                                                        ),
                                                        html.Tbody(
                                                            [
                                                                html.Tr(
                                                                    id="table-row-2",
                                                                    style={
                                                                        "border-left": "1px solid #ddd",
                                                                        "border-top": "1px solid #ddd"
                                                                    },
                                                                )
                                                            ]
                                                        ),
                                                        html.Tbody(
                                                            [
                                                                html.Tr(
                                                                    id="table-row-3",
                                                                    style={
                                                                        "border-left": "1px solid #ddd",
                                                                        "border-top": "1px solid #ddd"
                                                                    },
                                                                )
                                                            ]
                                                        ),
                                                    ]
                                                )
                                            ],
                                            style={
                                                "borderCollapse": "collapse",
                                                "width": "100%",
                                            },
                                        ),
                                    ],
                                    style={"border": "1px solid #ddd"},
                                ),
                            ],
                            id="dynamic-1",
                            style={
                                "--bs-gutter-x": "1.5rem",
                                "font-size": "13px",
                                "off-set": 4,
                                "color": "#000",
                                "padding-top": "10px",
                            },
                        ),
                    ),
                ),
            ]
        ),
        html.P(),
        html.Div(
            id="mapmessage",
            style={"color": "#FFFFFF", "fontSize": "20px", "marginTop": "-25px"},
        ),
        html.Br(),
        html.H5("What's On The Map?"),
        html.Hr(),
        html.P(
            """To understand this these two maps more thoroughly, a few things that are important to note are:"""
        ),
        dcc.Markdown(
            """
              1. There are three layers to the map that divide up the committees by party affiliation (on the top right corner of the map the results could be filtered through checking or unchecking each box).

              2. The backdrop layer displays the sum amount of funds raised for each state and the data could be displayed by hovering over each state boundary.

              3. The slider manipulates the committees to display by how much funds they have raised and the amount is indicated by the size of the colored dot (the more the bigger).

              4. The color of the dots/circles indicates the party affiliation of each committee.

              5. The stats that are right next to the slider indicate # PACs, average/total raise and spent (by party affiliation) for all the committees that fall into the sliding range.

             """
        ),
        html.Br(),
        html.H5("Some Other Important Info Stats"),
        html.Hr(),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dcc.Markdown(f""" - Total Raised by Party (Top 5): """),
                        dash_table.DataTable(
                            table1.to_dict("records"),
                            [{"name": i, "id": i} for i in table1.columns],
                            id="descriptive_table",
                            is_focused=True,
                            style_cell={
                                "textAlign": "left",
                                "border": "1px solid #ddd",
                                "fontSize": 15,
                            },
                            style_header={
                                "backgroundColor": "#cfd8dc",
                                "color": "black",
                                "fontWeight": "bold",
                            },
                        ),
                    ]
                ),
                dbc.Col(
                    [
                        dcc.Markdown(f""" - Total Comms by State and Party (Top 5):"""),
                        dash_table.DataTable(
                            table2.to_dict("records"),
                            [{"name": i, "id": i} for i in table2.columns],
                            id="descriptive_table",
                            is_focused=True,
                            style_cell={
                                "textAlign": "left",
                                "border": "1px solid #ddd",
                                "fontSize": 15,
                            },
                            style_header={
                                "backgroundColor": "#cfd8dc",
                                "color": "black",
                                "fontWeight": "bold",
                            },
                        ),
                    ]
                ),
                dbc.Col(
                    [
                        dcc.Markdown(
                            f""" - Total Raised by State and Party (Top 5): """
                        ),
                        dash_table.DataTable(
                            table3.to_dict("records"),
                            [{"name": i, "id": i} for i in table3.columns],
                            id="descriptive_table",
                            is_focused=True,
                            style_cell={
                                "textAlign": "left",
                                "border": "1px solid #ddd",
                                "fontSize": 15,
                            },
                            style_header={
                                "backgroundColor": "#cfd8dc",
                                "color": "black",
                                "fontWeight": "bold",
                            },
                        ),
                    ]
                ),
            ]
        ),


    ],
    className="page1",
    id="page1-content",
    style=PAGE_STYLE,
)


@callback(
    Output("cand-names-row", "children"),
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
            "color": "#000",
        },
    ), dcc.Dropdown(res, id="names-dropdown", searchable=True, multi=True)


@callback(
    [
        Output("candidates-stats-marker", "children"),
        Output("candidates-individual-marker", "viewport"),
        Output("candidates-individual-marker", "children"),
        Output("table-row-1", "children"),
        Output("table-row-2", "children"),
        Output("table-row-3", "children"),
    ],
    [
        dash.dependencies.Input("pac-exp-filter", "value"),
        dash.dependencies.Input("state-dropdown", "value"),
        dash.dependencies.Input("names-dropdown", "value"),
        dash.dependencies.Input("parties-checkbox", "value"),
        dash.dependencies.Input("checkboxes", "value"),
    ],
)
def update_output(slider, state, cands, parties, stats):
    latLon = candidates[
        [
            "Party code",
            "Party affiliation",
            "Affiliated Committee Name",
            "Total receipts",
            "Total disbursements",
            "lat",
            "lon",
        ]
    ]

    row = {"REP": 0, "DEM": 1, "3RD": 2}
    col = {
        "Avg Raised": 0,
        "Avg Spent": 1,
        "Total Raised": 2,
        "Total Spent": 3,
        "# PACs": 4,
    }
    latLon = [tuple(i[1:]) for i in latLon.itertuples()]
    colors = ["blue", "red", "#ddd"]
    s_latlon = [
        states[states["state"] == "DC"]["latitude"].iloc[0],
        states[states["state"] == "DC"]["longitude"].iloc[0],
    ]
    groups = {"DEM": ("blue", []), "REP": ("red", []), "OTH": ("#ddd", [])}

    n_rep = 0
    n_dem = 0
    n_3rd = 0

    raised_rep = 0
    raised_dem = 0
    raised_3rd = 0

    spent_rep = 0
    spent_dem = 0
    spent_3rd = 0

    cms = []

    for code, pty, name, r, s, lat, lng in latLon:

        # number of PACs divided by party
        if str(name) != 'nan':
            if 28000000 < r <= 30000000:
                radius = 53
                opacity = 0.8
            elif 24000000 < r <= 28000000:
                radius = 50
                opacity = 0.5
            elif 21000000 < r <= 24000000:
                radius = 45
                opacity = 0.5
            elif 18000000 < r <= 21000000:
                radius = 40
                opacity = 0.4
            elif 15000000 < r <= 18000000:
                radius = 35
                opacity = 0.3
            elif 12000000 < r <= 15000000:
                radius = 30
                opacity = 0.3
            elif 9000000 < r <= 12000000:
                radius = 20
                opacity = 0.3
            elif 6000000 < r <= 9000000:
                radius = 15
                opacity = 0.3
            elif 3000000 < r <= 6000000:
                radius = 10
                opacity = 0.3
            else:
                radius = 5
                opacity = 0.2

            if r > slider and r < slider + 2500000:
                cm = dl.CircleMarker(
                    center=[lat, lng],
                    color=colors[int(code) - 1],
                    opacity=0.8,
                    weight=1,
                    fillColor=colors[int(code) - 1],
                    fillOpacity=opacity + 0.25,
                    radius=radius,
                    children=[
                        dl.Tooltip(
                            children=html.Div(
                                children=[
                                    html.Div(["Committee Name: ", html.B(f"{name}")]),
                                    html.Div(["Election cycle: ", html.B("2022")]),
                                    html.Div(["Total Raised (YTD2022): ", html.B(f"${r}")]),
                                    html.Div(["Total Spent (YTD2022): ", html.B(f"${s}")]),
                                ],
                                style={
                                    "width": "250px",
                                    "backgroundColor": "#fff",
                                    "borderRadius": "5px",
                                    "padding": "10px",
                                    "boxShadow": "0 2px 5px rgba(0, 0, 0, 0.3)",
                                    "color": "#333",
                                    "whiteSpace": "pre-wrap",
                                    "font-size": "14px",
                                },
                            ),
                        )
                    ],
                )

                if pty != "REP" and pty != "DEM":
                    n_3rd += 1
                    raised_3rd += r
                    spent_3rd += s
                    groups["OTH"][1].append(cm)
                else:
                    if pty == "REP":
                        n_rep += 1
                        raised_rep += r
                        spent_rep += s
                    else:
                        n_dem += 1
                        raised_dem += r
                        spent_dem += s
                    groups[pty][1].append(cm)
                cms.append(cm)

    avg_r_rep = round(raised_rep / n_rep, 1) if n_rep != 0 else 0
    avg_r_dem = round(raised_dem / n_dem, 1) if n_dem != 0 else 0
    avg_r_3rd = round(raised_3rd / n_3rd, 1) if n_3rd != 0 else 0

    avg_s_rep = round(spent_rep / n_rep, 1) if n_rep != 0 else 0
    avg_s_dem = round(spent_dem / n_dem, 1) if n_dem != 0 else 0
    avg_s_3rd = round(spent_3rd / n_3rd, 1) if n_3rd != 0 else 0

    template = [
        html.Td(
            0,
            id="rep-0",
            style={
                "vertical-align": "middle",
                "width": "15%",
            },
        ),
        html.Td(
            0,
            id="rep-1",
            style={
                "vertical-align": "middle",
                "width": "13%",
            },
        ),
        html.Td(
            0,
            id="rep-2",
            style={
                "vertical-align": "middle",
                "width": "15%",
            },
        ),
        html.Td(
            0,
            id="rep-3",
            style={
                "vertical-align": "middle",
                "width": "17%",
            },
        ),
        html.Td(
            0,
            id="rep-4",
            style={
                "vertical-align": "middle",
                "width": "1%",
            },
        ),
    ]

    template1 = [
        html.Td(
            0,
            id="rep-0",
            style={
                "vertical-align": "middle",
                "width": "13.5%%",
            },
        ),
        html.Td(
            0,
            id="rep-1",
            style={
                "vertical-align": "middle",
                "width": "13%",
            },
        ),
        html.Td(
            0,
            id="rep-2",
            style={
                "vertical-align": "middle",
                "width": "15%",
            },
        ),
        html.Td(
            0,
            id="rep-3",
            style={
                "vertical-align": "middle",
                "width": "17%",
            },
        ),
        html.Td(
            0,
            id="rep-4",
            style={
                "vertical-align": "middle",
                "width": "1%",
            },
        ),
    ]
    template2 = [
        html.Td(
            0,
            id="dem-0",
            style={
                "vertical-align": "middle",
                "width": "13.5%",
            },
        ),
        html.Td(
            0,
            id="dem-1",
            style={
                "vertical-align": "middle",
                "width": "13%",
            },
        ),
        html.Td(
            0,
            id="dem-2",
            style={
                "vertical-align": "middle",
                "width": "15%",
            },
        ),
        html.Td(
            0,
            id="dem-3",
            style={
                "vertical-align": "middle",
                "width": "17%",
            },
        ),
        html.Td(
            0,
            id="dem-4",
            style={
                "vertical-align": "middle",
                "width": "1%",
            },
        ),
    ]
    template3 = [
        html.Td(
            0,
            id="3rd-0",
            style={
                "vertical-align": "middle",
                "width": "13.5%",
            },
        ),
        html.Td(
            0,
            id="3rd-1",
            style={
                "vertical-align": "middle",
                "width": "13%",
            },
        ),
        html.Td(
            0,
            id="3rd-2",
            style={
                "vertical-align": "middle",
                "width": "15%",
            },
        ),
        html.Td(
            0,
            id="3rd-3",
            style={
                "vertical-align": "middle",
                "width": "17%",
            },
        ),
        html.Td(
            0,
            id="3rd-4",
            style={
                "vertical-align": "middle",
                "width": "1%",
            },
        ),
    ]

    rep = [avg_r_rep, avg_s_rep, raised_rep, spent_rep, n_rep]
    dem = [avg_r_dem, avg_s_dem, raised_dem, spent_dem, n_dem]
    trd = [avg_r_3rd, avg_s_3rd, raised_3rd, spent_3rd, n_3rd]

    row1 = [i for i in template]
    row2 = [i for i in template]
    row3 = [i for i in template]

    for i in range(5):
        if i != 4:
            v1 = format_currency(round(rep[i], 1))
            v2 = format_currency(round(dem[i], 1))
            v3 = format_currency(round(trd[i], 1))
        else:
            v1 = f" {round(rep[i], 1)}"
            v2 = f" {round(dem[i], 1)}"
            v3 = f"  {round(trd[i], 1)}"
        style = row1[i].style
        td = html.Td(children=v1 , id=f"rep-{i}", style=style)
        row1[i] = td

        style = row2[i].style
        td = html.Td(children=v2, id=f"dem-{i}", style=style)
        row2[i] = td

        style = row3[i].style
        td = html.Td(children=v3, id=f"3rd-{i}", style=style)
        row3[i] = td


    row1.append(rep[-1])
    row2.append(dem[-1])
    row3.append(trd[-1])

    tmp = [template1, template2, template3]
    rows = [row1, row2, row3]

    if parties:
        for i in parties:
            if stats:
                for j in stats:
                    tmp[row[i]][col[j]] = rows[row[i]][col[j]]
        row1, row2, row3 = tmp

    else:
        for i in range(3):
            if stats:
                for j in stats:
                    tmp[i][col[j]] = rows[i][col[j]]
        row1, row2, row3 = tmp


    data = [
        dl.TileLayer(url=style_url,
             attribution=attribution),
        dl.LayersControl(
            [
                dl.Pane(
                    name="US-Boundaries",
                    children=[
                        dl.BaseLayer(
                            name="US-Boundaries", children=choropleth1, checked=True
                        )
                    ],
                ),
                dl.Pane(
                    name="PAC-Data",
                    children=[
                        dl.Overlay(
                            dl.LayerGroup(groups["DEM"][1]), name="DEM", checked=True
                        ),
                        dl.Overlay(
                            dl.LayerGroup(groups["REP"][1]), name="REP", checked=True
                        ),
                        dl.Overlay(
                            dl.LayerGroup(groups["OTH"][1]), name="3RD", checked=True
                        ),
                    ],
                ),
            ]
        ),
    ]

    second_map = data.copy()
    second_map.remove(second_map[1])
    second_map.insert(
        1,
        dl.LayersControl(
            [
                dl.Pane(
                    name="US-Boundaries",
                    children=[
                        dl.BaseLayer(
                            name="US-Boundaries", children=choropleth2, checked=True
                        )
                    ],
                ),
                dl.Pane(
                    name="PAC-Data",
                    children=[
                        dl.Overlay(
                            dl.LayerGroup(groups["DEM"][1]), name="DEM", checked=True
                        ),
                        dl.Overlay(
                            dl.LayerGroup(groups["REP"][1]), name="REP", checked=True
                        ),
                        dl.Overlay(
                            dl.LayerGroup(groups["OTH"][1]), name="3RD", checked=True
                        ),
                    ],
                ),
            ]
        ),
    )
    if state:
        if states[states["name"] == state]["capital"].iloc[0]:
            s_latlon = [
                states[states["name"] == state]["latitude_c"].iloc[0],
                states[states["name"] == state]["longitude_c"].iloc[0],
            ]
        else:
            s_latlon = [
                states[states["name"] == state]["latitude"].iloc[0],
                states[states["name"] == state]["longitude"].iloc[0],
            ]
        if cands:
            pins = []
            for cand in cands:
                row = candidates[candidates["Candidate name"] == cand]
                latlng = [row["lat"].iloc[0], row["lon"].iloc[0]]
                m = dl.Marker(
                    position=latlng,
                    children=[
                        dl.Tooltip(
                            children=html.Div(
                                        children=[
                                            html.Div(
                                                [
                                                    "Committee Name: ",
                                                    html.B(f'{row["Affiliated Committee Name"].iloc[0]}'),
                                                ]
                                            ),
                                            html.Div(
                                                [
                                                    "Election cycle: ",
                                                    html.B("2022"),
                                                ]
                                            ),
                                            html.Div(
                                                [
                                                    "Total Raised (YTD2022): ",
                                                    html.B(f'${row["Total receipts"].iloc[0]}'),
                                                ]
                                            ),
                                            html.Div(
                                                [
                                                    "Total Spent (YTD2022): ",
                                                    html.B(f'${row["Total disbursements"].iloc[0]}'),
                                                ]
                                            ),
                                        ],
                                style={
                                    "width": "250px",
                                    "backgroundColor": "#fff",
                                    "borderRadius": "5px",
                                    "padding": "10px",
                                    "boxShadow": "0 2px 5px rgba(0, 0, 0, 0.3)",
                                    "color": "#333",
                                    "whiteSpace": "pre-wrap",
                                },
                            ),
                        )
                    ],
                )
                pins.append(m)
            second_map.append(dl.Pane(name="Individual-pin", children=pins))
            s_latlon = [row["lat"].iloc[0], row["lon"].iloc[0]]
    else:
        if cands:
            pins = []
            for cand in cands:
                row = candidates[candidates["Candidate name"] == cand]
                latlng = [row["lat"].iloc[0], row["lon"].iloc[0]]
                m = dl.Marker(
                    position=latlng,
                    children=[
                        dl.Tooltip(
                            children=html.Div(
                                children=[
                                    html.Div(
                                        [
                                            "Committee Name: ",
                                            html.B(
                                                f'{row["Affiliated Committee Name"].iloc[0]}'
                                            ),
                                        ]
                                    ),
                                    html.Div(
                                        [
                                            "Election cycle: ",
                                            html.B("2022"),
                                        ]
                                    ),
                                    html.Div(
                                        [
                                            "Total Raised (YTD2022): ",
                                            html.B(f'${row["Total receipts"].iloc[0]}'),
                                        ]
                                    ),
                                    html.Div(
                                        [
                                            "Total Spent (YTD2022): ",
                                            html.B(
                                                f'${row["Total disbursements"].iloc[0]}'
                                            ),
                                        ]
                                    ),
                                ],
                                style={
                                    "width": "250px",
                                    "backgroundColor": "#fff",
                                    "borderRadius": "5px",
                                    "padding": "10px",
                                    "boxShadow": "0 2px 5px rgba(0, 0, 0, 0.3)",
                                    "color": "#333",
                                    "whiteSpace": "pre-wrap",
                                    "font-size": "14px",
                                },
                            ),
                        )
                    ],
                )
                pins.append(m)
            second_map.append(dl.Pane(name="Individual-pin", children=pins))
            s_latlon = [row["lat"].iloc[0], row["lon"].iloc[0]]

    return (
        data,
        dict(center=s_latlon, zoom=7, transition="flyTo"),
        second_map,
        row1,
        row2,
        row3,
    )


@callback(Output("info1", "children"), dash.dependencies.Input("geojson1", "hoverData"))
def info_hover(feature):
    return get_info(feature)


@callback(Output("info2", "children"), dash.dependencies.Input("geojson2", "hoverData"))
def info_hover(feature):
    return get_info(feature)
