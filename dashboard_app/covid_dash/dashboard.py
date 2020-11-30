"""Instantiate a Dash app."""
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_table
import plotly.express as px
import json

from .data import create_dataframe

def init_dashboard(server):
    """Create a Plotly Dash dashboard."""
    dash_app = dash.Dash(
        server=server,
        routes_pathname_prefix="/",
        external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css', dbc.themes.BOOTSTRAP, dbc.themes.GRID,
                              dbc.themes.CERULEAN]

    )
    # Load DataFrame
    df = create_dataframe()

    # Load county geo data.
    with open('data/geojson-counties-fips.json') as f:
        counties = json.load(f)

    # Create dashboard.
    fig = px.choropleth(df, geojson=counties, locations='fips', color='Tier',
                        # color_continuous_scale="Viridis",
                        color_discrete_map={'1': 'rgb(128,0,128)', '2': 'rgb(255,0,0)', '3': '	(255,165,0)',
                                            '4': '(255,255,0)'},
                        range_color=(1, 4),
                        animation_frame="Date",
                        hover_data=["Tier", "Cases", "R0"],
                        scope="usa",
                        labels={'Tier': 'County Tier', 'Cases': 'New cases', "R0": "Transmission rate"}
                        )
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    fig["layout"].pop("updatemenus")

    dash_app.layout = html.Div(children=[
        html.H1(children=''),
        dcc.Graph(
            id='covid-prediction-graph',
            figure=fig
        )
    ])
    return dash_app.server


def create_data_table(df):
    """Create Dash datatable from Pandas DataFrame."""
    table = dash_table.DataTable(
        id="database-table",
        columns=[{"name": i, "id": i} for i in df.columns],
        data=df.to_dict("records"),
        sort_action="native",
        sort_mode="native",
        page_size=300,
    )
    return table
