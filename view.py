import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from datetime import datetime as dt
import plotly.graph_objects as go
import seaborn as sns

from comm_data import *


def scatterplot_efficiency_date(dt, com, eff):

    dt['DateTime'] = pd.to_datetime(dt['DateTime'])

    dt = dt.sort_values(by=['DateTime'])

    dt = dt[["DateTime", "Efficiency"]].reset_index(drop=True)

    dt = dt[(dt['Efficiency'] > 0)]


    dt_com = com
    min_dt = dt["DateTime"].min()
    max_dt = dt["DateTime"].max()
    
    min_com = dt_com[eff].min()
    max_com = dt_com[eff].max()
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
                     x=dt["DateTime"],
                     y=dt["Efficiency"],
                     name="Actual Data",
                     mode='markers',
                     marker_color='lightskyblue',
                     marker_size=8,
                     marker_line_width=1
                     ))
    fig.add_shape(type = 'line', x0=min_dt, y0=min_com, x1=max_dt, y1=min_com, line= dict(color='green'), xref='x', yref='y', )
    
    fig.add_shape(type = 'line', x0=min_dt, y0=max_com, x1=max_dt, y1=max_com, line= dict(color='green'), xref='x', yref='y')
    
    min_text = 'Minimum: ' + str(min_com)
    max_text = 'Maximum: ' + str(max_com)
    
    fig.add_annotation(text=min_text,xanchor='left',x=max_dt,y=min_com,arrowhead=1,showarrow=False)
    fig.add_annotation(text=max_text,xanchor='left',x=max_dt,y=max_com,arrowhead=1,showarrow=False)
    
  
    fig.update_layout(title_text="Hourly Efficiency",title_x=0, xaxis_title="Date", yaxis_title="Efficiency",
                        margin=dict(l=0,r=10,b=10,t=30),
                        legend=dict(orientation="h",yanchor="bottom",y=0.9,xanchor="right",x=0.99))

    st.plotly_chart(fig, use_container_width=True)  


def scatterplot_discharge_power(dt, com):

    dt['DateTime'] = pd.to_datetime(dt['DateTime'])

    # dt = dt.sort_values(by=['DateTime'])

    dt = dt[["Discharge", "Power Output", "Efficiency"]
            ].reset_index(drop=True)

    dt = dt[(dt['Efficiency'] > 0)]
    dt = dt.sort_values(by=['Power Output'])
    dt_com = com
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
                     x=dt["Power Output"],
                     y=dt["Discharge"],
                     name="Actual Data",
                     mode='markers',
                     marker_color='lightskyblue',
                     marker_size=8,
                     marker_line_width=1
                     ))
    
    
    fig.add_trace(go.Scatter(
                     x=dt_com["Power Output"],
                     y=dt_com['Unit Flow'],
                     name="Commission Data",
                     mode='markers',
                     marker_symbol='x-dot',
                     marker_color='rgba(152, 0, 0, .8)',
                     marker_size=10,
                     marker_line_width=2
                     
                     ))
   
    

    fig.update_layout(title_text="Disharge and Efficiency",title_x=0, xaxis_title="Power Output", yaxis_title="Discharge (cms)",
                        margin=dict(l=0,r=10,b=10,t=30), 
                        legend=dict(orientation="h",yanchor="bottom",y=0.9,xanchor="right",x=0.99))

    st.plotly_chart(fig, use_container_width=True)  


def scatterplot_theoretical_actual_power(dt):

    dt['DateTime'] = pd.to_datetime(dt['DateTime'])

   

    dt = dt[["Theoretical Power", "Power Output", "Efficiency"]
            ].reset_index(drop=True)

    dt = dt[(dt['Efficiency'] > 0)]
    dt = dt.sort_values(by=['Power Output'])
    
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
                     x=dt["Power Output"],
                     y=dt["Theoretical Power"],
                     mode='markers',
                     marker_color='lightskyblue',
                     marker_size=8,
                     marker_line_width=1
                     ))
   
    
    fig.update_layout(xaxis_title="Actual Power", yaxis_title="Theoretical Power",
                      yaxis_zeroline=False, xaxis_zeroline=False,
                      width=912
                  
    )

    fig.update_layout(title_text="Theoretical Power and Actual Power",title_x=0, xaxis_title="Actual Power", yaxis_title="Theoretical Power",
                        margin=dict(l=0,r=10,b=10,t=30), 
                        legend=dict(orientation="h",yanchor="bottom",y=0.9,xanchor="right",x=0.99))

    st.plotly_chart(fig, use_container_width=True)  


def scatterplot_discharge_efficiency(dt):

    dt['DateTime'] = pd.to_datetime(dt['DateTime'])

   

    dt = dt[["Discharge", "Efficiency"]
            ].reset_index(drop=True)

    dt = dt[(dt['Efficiency'] > 0)]
    dt = dt.sort_values(by=['Efficiency'])
    
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
                     x=dt["Discharge"],
                     y=dt["Efficiency"],
                     mode='markers',
                     marker_color='lightskyblue',
                     marker_size=8,
                     marker_line_width=1
                     ))
   
    
    fig.update_layout(title_text="Discharge and Efficiency",title_x=0, xaxis_title="Discharge", yaxis_title="Efficiency",
                        margin=dict(l=0,r=10,b=10,t=30), 
                        legend=dict(orientation="h",yanchor="bottom",y=0.9,xanchor="right",x=0.99))
    st.plotly_chart(fig, use_container_width=True)  


def scatterplot_efficiency_power(dt, com, eff):

    dt['DateTime'] = pd.to_datetime(dt['DateTime'])

    # dt = dt.sort_values(by=['DateTime'])
    
    dt = dt[["Discharge", "Power Output", "Efficiency"]
            ].reset_index(drop=True)

    dt = dt[(dt['Efficiency'] > 0)]
    dt = dt.sort_values(by=['Power Output'])

    dt_com = com
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
                     x=dt["Power Output"],
                     y=dt["Efficiency"],
                     name="Actual Data",
                     mode='markers',
                     marker_color='lightskyblue',
                     marker_size=8,
                     marker_line_width=1
                     ))
    
    
    fig.add_trace(go.Scatter(
                     x=dt_com["Power Output"],
                     y=dt_com[eff],
                     name="Commission Data",
                     mode='markers',
                     marker_symbol='x-dot',
                     marker_color='rgba(152, 0, 0, .8)',
                     marker_size=10,
                     marker_line_width=2
                     ))

    
    fig.update_layout(title_text="Efficiency and Power Output",title_x=0, xaxis_title="Power Output", yaxis_title="Efficiency",
                        margin=dict(l=0,r=10,b=10,t=30), 
                        legend=dict(orientation="h",yanchor="bottom",y=0.9,xanchor="right",x=0.99))

    st.plotly_chart(fig, use_container_width=True)  


def view_heatmap(dataframe):
    # dt = dt[["Discharge", "Velocity", "Pressure Head", "Velocity Head",
    #          "Net Head", "Theoretical Power", "Power Output", "Efficiency"]].reset_index(drop=True)

    df = dataframe.drop(columns=["DateTime","Plant", "Unit"], axis=1)

    df = df[(df['Efficiency'] > 0)]
    
    corr = df.corr()

    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=(10, 5))

    cmap = sns.diverging_palette(230, 15, as_cmap=True)

    ax = sns.heatmap(corr, mask=mask, cmap=cmap,  center=0, annot=True,
                square=False, linewidths=.5, cbar_kws={"shrink": .5})

    st.write(fig)

def view_histogram(dataframe): ## Efficiency, Power Output  Discharge
    df = dataframe[(dataframe['Efficiency'] > 0)]

    h1, h2, h3  = st.columns((1,1,1))
    
    fig = px.histogram(df, x="Efficiency", nbins=20)

    fig.update_layout(title_text="Efficiency Histogram",title_x=0,
                        margin=dict(l=0,r=0,b=15,t=100))

    fig_1 = px.histogram(df, x="Power Output", nbins=20)
    
    fig_1.update_layout(title_text="Power Output Histogram",title_x=0,
                        margin=dict(l=0,r=0,b=15,t=100))

    fig_2 = px.histogram(df, x="Discharge", nbins=20)
    
    fig_2.update_layout(title_text="Discharge Histogram",title_x=0,
                        margin=dict(l=0,r=0,b=15,t=100))                                           

    h1.plotly_chart(fig, use_container_width=True)
    h2.plotly_chart(fig_1, use_container_width=True)
    h3.plotly_chart(fig_2, use_container_width=True)


def view_3d_power_efficiency_discharge(dataframe):
    df = dataframe[(dataframe['Efficiency'] > 0)]
    df['Month'] = pd.to_datetime(df['DateTime']).dt.strftime('%B')
  

    fig = px.scatter_3d(df, x='Discharge', y='Power Output', z='Efficiency', color='Month')

    fig.update_layout(title_text="Efficiency, Power Output and Discharge",title_x=0,
                        margin=dict(l=0,r=0,b=15,t=100), 
                        legend=dict(orientation="h",yanchor="bottom",y=0.9,xanchor="right",x=0.99))

    st.plotly_chart(fig, use_container_width=True) 

