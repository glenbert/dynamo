import streamlit as st
import pandas as pd
from datetime import datetime as dt


@st.cache
def load_data(file):
    if file is not None:
        data = pd.read_csv(file)
        def dtm(x): return dt.strptime(str(x), '%d/%m/%Y %H:%M')

        # data['Date'] = pd.to_datetime(data['Date'])
        data["DateTime"] = data["DateTime"].apply(dtm)
        return data


def get_selected_data(dt, plant, unit):

    df = dt[(dt["Plant"] == plant)
            & (dt["Unit"] == unit)]

    data = df[(df['Efficiency'] > 0)].reset_index(drop=True)

    return data