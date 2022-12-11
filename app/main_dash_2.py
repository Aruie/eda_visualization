from dash.dependencies import Input, Output
from fastapi import FastAPI
from fastapi.middleware.wsgi import WSGIMiddleware
from dash import html, dcc, Dash, dash_table
from fastapi import FastAPI, APIRouter, Query, HTTPException, Request
from pydantic import BaseModel
from typing import Optional, Any
from pathlib import Path
from fastapi.templating import Jinja2Templates
import pandas as pd

import plotly.express as px
import numpy as np


BASE_PATH = Path(__file__).resolve().parent
TEMPLATES = Jinja2Templates(directory=str(BASE_PATH / "templates"))


file_name = './static/data/titanic.csv'
df = pd.read_csv(file_name)

# Create the Dash application, make sure to adjust requests_pathname_prefx
app_dash = Dash(__name__, requests_pathname_prefix='/dash/')


def make_components(style={'background-color': 'rgb(250, 250, 150)', 'float': 'left'}):
    components = []
    components.append(
        dcc.RadioItems(['Original', 'Standardization', 'Normalization'], 'Original',
                       id='normalization'))

    components.append(
        dcc.RadioItems(['Original', 'Log Scale'], 'Original', id='scale'))

    components.append(
        dcc.RadioItems(['Original', 'Fill Zero', 'Fill Mean'], 'Original',
                       id='missing'))

    return html.Div(components, id='components', style=style)




def make_header(df, column):
    df['mod'] = df[column]

    na_cnt = df[column].isna().sum()
    length = len(df[column])
    cardinality = len(df[column].unique())


    layout = html.Div([

        dcc.Markdown(f"""
        ## Summary
        - Length : {length} 
        - NA count : {na_cnt} ({na_cnt / length * 100:.2f}%)
        - Cardinality : {cardinality} ({cardinality / length * 100:.2f}%)
        """, 
            style={'background-color': 'rgb(250, 250, 150)', 'float': 'left', 'width': '50%'}),
    
        make_components(style={'float': 'left', 'width': '50%', 'background-color': 'rgb(250, 250, 150)', 'display':'flex', 'flex-direction':'column', 'justify-content':'space-around'})
        ], style={'float': 'top', 'display':'flex', 'flex-direction':'row', 'max-width':'1280px'})
    return layout





def make_layout(df, column, mods=[]):
    df['mod'] = df[column]

    if 'fill_zero' in mods:
        df['mod'] = df['mod'].fillna(0)
    elif 'fill_mean' in mods:
        df['mod'] = df['mod'].fillna(df['mod'].mean())

    if 'log_scaled' in mods:
        df['mod'] = np.log(df['mod'] + 1)

    if 'standardization' in mods:
        m, s = df['mod'].describe()[['mean', 'std']]
        df['mod'] = (df['mod'] - m) / s
    elif 'normalization' in mods:
        m, s = df['mod'].describe()[['min', 'max']]
        df['mod'] = (df['mod'] - m) / (s - m)

    components = []

    components.append(
        html.Div([
            dcc.Graph(figure=px.histogram(df, x=column, marginal='box', color='Survived',
                                          barmode='overlay', nbins=20, title='Original Distribution'),
                      style={'float': 'left', 'background-color': 'rgb(230, 230, 230)', 'width':'50%'}),

            dcc.Graph(figure=px.histogram(df, x='mod', color='Survived', marginal='box',
                                          barmode='overlay', nbins=20, title='Modified Distribution'),
                      style={'float': 'left', 'background-color': 'rgb(230, 230, 230)','width':'50%'}),
        ], style={'float': 'top', 'display' : 'flex', 'flex-direction':'row', 'max-width':'1280px'})
    )

    return html.Div(components, id='layouts')


column = df.columns[0]

@app_dash.callback(Output('layouts', 'children'),
                   [Input('normalization', 'value'),
                    Input('scale', 'value'),
                    Input('missing', 'value'),
                    Input('column', 'value')])
def update_layout(normalization, scale, missing, column):
    mods = []
    if normalization == 'Standardization':
        mods.append('standardization')
    elif normalization == 'Normalization':
        mods.append('normalization')

    if scale == 'Log Scale':
        mods.append('log_scaled')

    if missing == 'Fill Zero':
        mods.append('fill_zero')
    elif missing == 'Fill Mean':
        mods.append('fill_mean')

    if column is None:
        column = df.columns[0]
        
    return make_layout(df, column, mods)


app_dash.layout = html.Div(children=[
    dcc.Dropdown(df.columns.tolist(), id='column', style={'max-width':'1280px'}),
    make_header(df, column),
    make_layout(df, column),
])


# Now create your regular FASTAPI application
app = FastAPI()
# Now mount you dash server into main fastapi application
app.mount("/dash", WSGIMiddleware(app_dash.server))
