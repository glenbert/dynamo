import streamlit as st
from PIL import Image
from view import *
from model import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from datetime import datetime as dt
import plotly.graph_objects as go


st.set_page_config(page_title='DyNaMo',  layout='wide', page_icon=':chart_with_upwards_trend:')
img_logo = Image.open('hedcor-logo.png')
st.image(img_logo, width=200)
st.markdown("HEDCOR - Reliability and Asset Management")


st.markdown("#### Dynamic Turbine Performance Monitoring Platform")
    
with st.expander("Upload Data"):
    file = st.file_uploader("Choose CSV file", type="csv")


def main():
    
    if file is not None:
        
        data = load_data(file)

        PLANT = data['Plant'].unique()
        UNIT = data['Unit'].unique()

        
        m1, m2, m3, m4, m5 = st.columns((1,1,1,1,1))

        PLANT_SELECTED = m1.selectbox('Select plant', PLANT)
        UNIT_SELECTED = m2.selectbox('Select Unit', UNIT)
        option_eff = m3.selectbox('Select Type of Efficiency',('Turbine Guaranted Eff.', 'Generator Guaranted Eff.', 'Overall Eff.'))

        if UNIT_SELECTED == "Unit 1":    
          unt = df_comm_unit_1
          unt = df_comm_unit_1[(df_comm_unit_1["Plant"]== PLANT_SELECTED)]
        else:
          unt = df_comm_unit_2
          unt = df_comm_unit_2[(df_comm_unit_1["Plant"]== PLANT_SELECTED)]
        
        df = get_selected_data(data, PLANT_SELECTED, UNIT_SELECTED)

        st.markdown('### Tubine - Generator Commissioning Result')
        unt_ = unt.drop('Plant', axis=1)
        st.dataframe(unt_)
        
        
        # st.markdown('### Efficiency and Power Output')
        
        scatterplot_efficiency_power(df, unt, option_eff)

        view_histogram(df)
        
        # st.markdown('### Discharge and Power Output')
        scatterplot_discharge_power(df, unt)

        # st.markdown('### Theoretical Power and Power Output')
        scatterplot_theoretical_actual_power(df)
        
        # st.markdown('### Discharge and Efficiency')
        scatterplot_discharge_efficiency(df)

        # st.markdown('### Hourly Efficiency')
        scatterplot_efficiency_date(df, unt, option_eff)

        view_3d_power_efficiency_discharge(df)
        
        view_heatmap(df)


if __name__ == "__main__":
    main()
