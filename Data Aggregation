"""
This module contains all US-specific data loading and data cleaning routines.
"""
import glob
import os
import pandas as pd

idx = pd.IndexSlice


def process_covidtracking_data(cdbc: pd.DataFrame, tpr_state: pd.DataFrame, run_date: pd.Timestamp):
    """ Processes raw COVIDTracking data by county level."""
    cdbc["date"] = pd.to_datetime(cdbc["date"], format="%Y-%m-%d")
    cdbc = cdbc.sort_values("date")
    cdbc = cdbc.rename(columns={"state": "region"})
    cdbc["positive"] = cdbc.groupby(by=["region", "county"])["cases"].diff()
    tpr_state["date"] = pd.to_datetime(tpr_state["date"], format="%m-%d-%Y")
    tpr_state = tpr_state.sort_values("date")
    tpr_state["Confirmed"] = tpr_state.groupby("Province_State")["Confirmed"].diff()
    tpr_state["People_Tested"] = tpr_state.groupby("Province_State")["People_Tested"].diff()
    tpr_state["positivity_rate"] = tpr_state["Confirmed"] / tpr_state["People_Tested"]
    tpr_state = tpr_state.rename(columns={"Province_State": "region"})
    tpr_state = tpr_state[["date", "region", "positivity_rate", "Confirmed", "People_Tested"]]
    tpr_state.head(100)
    data = pd.merge(cdbc, tpr_state, on=['date', 'region'], how='right')
    data = data[data['positivity_rate'].notna()]
    data['total'] = (data['positive'] / data['positivity_rate']).round(0)
    data = data.drop(columns=['fips', 'positivity_rate'])
    data = data[data['total'].notna()]
    data['region'] = data['region'] + "_" +  data['county']
    data = data[['region','date', 'positive', 'total']]
    data = data.set_index(["region","date"]).sort_index()
    return data.loc[idx[:, :(run_date.to_pydatetime() - pd.DateOffset(1))], ["positive", "total"]]



def get_raw_covidtracking_data():
    """ Gets the covid tracking data by county"""
    # Gets the current daily CSV from NY Times
    url = "https://github.com/nytimes/covid-19-data/blob/master/us-counties.csv"
    cdbc = pd.read_csv('../data/us-counties.csv', index_col=None, header=0,
                       error_bad_lines=False)
    # Get test positivity rate by state.
    path = r'../data/tpr'
    all_files = glob.glob(path + "/*.csv")
    li = []
    for filename in all_files:
        df = pd.read_csv(filename, index_col=None, header=0)
        df['date'] = os.path.basename(filename)[:-4]  # Removing ".csv" last 4 characters from filename.
        li.append(df)

    tpr_state = pd.concat(li, axis=0, ignore_index=True)
    #tpr_state.to_csv('/home/pravalli/PycharmProjects/covid-model/data/Dataoutput.csv', sep='\t')
    return cdbc, tpr_state


def get_and_process_covidtracking_data(run_date: pd.Timestamp):
    """ Helper function for getting and processing COVIDTracking data at once """
    cdbc, tpr_state = get_raw_covidtracking_data()
    data = process_covidtracking_data(cdbc, tpr_state, run_date)
    return data
