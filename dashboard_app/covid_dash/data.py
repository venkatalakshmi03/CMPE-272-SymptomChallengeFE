"""Prepare data for Plotly Dash."""
import numpy as np
import pandas as pd


def create_dataframe():
    """Create Pandas DataFrame from local CSV."""
    # df = pd.read_csv('data/owid-covid-data.csv')
    # df = df[df.location != 'World']
    # # Sort df by date
    # df = df.sort_values(by=['date'])
    df = pd.read_csv("data/covid_tier_demo_data.csv",
                     dtype={"fips": str})
    return df
