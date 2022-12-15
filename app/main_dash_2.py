from dash.dependencies import Input, Output, State
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


# file_name = './static/data/titanic.csv'
file_name = './static/data/train.csv'

df = pd.read_csv(file_name).sample(1000)

env_dict = {
    'Column Type': 'Numerical',
    'Normalization': 'Original',
    'Scale': 'Original',
    'Missing': 'Original',
    'lambda_exp': '',
}

env_save = {



}


# Create the Dash application, make sure to adjust requests_pathname_prefx
app_dash = Dash(__name__, requests_pathname_prefix='/dash/')


# Header
def make_top():
    div = html.Div(children=[
        dcc.Dropdown(df.columns.tolist(), df.columns[0], id='column', style={
                     'max-width': '1024px'}),

        html.Div(make_layout(df, column), id='layouts',
                 style={'float': 'left', 'width': '100%'}),
    ])
    return div


def make_components():
    
    def make_radio(name, items, id):
        tmp = []
        tmp.append(dcc.Markdown(
            name, style={'width': '20%', 'text-align': 'top'}))
        tmp.append(dcc.RadioItems(items, env_dict[name], id=id, style={
                   'vertical-align': 'middle'}))
        return html.Div(tmp, style={'float': 'top', 'width': '100%', 'height': '20%',
                                    'display': 'flex', 'flex-direction': 'row', 'border': '1px solid black'})

    divs = []
    divs.append(make_radio('Column Type', ['Numerical', 'Categorical'], id='column_type'))
    divs.append(make_radio('Normalization', [
                'Original', 'Standardization', 'Normalization'],  id='normalization'))
    divs.append(make_radio('Scale', ['Original', 'Log Scale'],  id='scale'))
    divs.append(make_radio(
        'Missing', ['Original', 'Fill Zero', 'Fill Mean'], id='missing'))

    tmp = []
    tmp.append(dcc.Markdown('Lambda x', style={
               'width': '20%', 'text-align': 'top'}))
    tmp.append(dcc.Textarea(id='lambda_exp', rows=1,
               style={'vertical-align': 'middle', 'width': '80%'}))
    divs.append(html.Div(tmp, style={'float': 'top', 'width': '100%', 'height': '20%',
                                     'display': 'flex', 'flex-direction': 'row', 'border': '1px solid black', 'background-color': 'rgb(250, 250, 150)'}))

    return html.Div(divs, style={'width': '50%', 'height': '190px', 'float': 'right',
                                 'display': 'flex', 'flex-direction': 'column', 'background-color': 'rgb(250, 250, 150)'})

def make_layout(df, column, col_type = 'Numerical'):
    div1 = []
    print(env_dict)
    div1.append(make_column_information(df, column))
    div1.append(make_components())
    
    div1 = html.Div(div1, style={'float': 'top', 'width': '100%', 'background-color': 'rgb(250, 250, 150)'})

    if col_type == 'Numerical':
        div = make_numeric_layout(df, column, env_dict)
    elif col_type == 'Categorical':
        div = make_categorical_layout(df, column, env_dict)

    return html.Div([div1, div], style={'float': 'top', 'width': '100%'})

def make_column_information(df, column):
    df['mod'] = df[column]

    na_cnt = df[column].isna().sum()
    length = len(df[column])
    cardinality = len(df[column].unique())

    component = dcc.Markdown(f"""
        ## Columns : {column}
        - Length : {length} 
        - NA count : {na_cnt} ({na_cnt / length * 100:.2f}%)
        - Cardinality : {cardinality} ({cardinality / length * 100:.2f}%)
        """,
                             style={'float': 'left', 'background-color': 'rgb(250, 250, 150)', 'width': '50%', 'height': '200px'})
    return component


def make_numeric_layout(df, column, env_dict):

    df['mod'] = df[column]

    if env_dict['Missing'] == 'Fill Zero':
        df['mod'] = df['mod'].fillna(0)
    elif env_dict['Missing'] == 'Fill Mean' :
        df['mod'] = df['mod'].fillna(df['mod'].mean())

    if env_dict['Scale'] == 'Log Scale' :
        df['mod'] = np.log(df['mod'] + 1)

    if env_dict['Normalization'] == 'Standardization':
        m, s = df['mod'].describe()[['mean', 'std']]
        df['mod'] = (df['mod'] - m) / s
    elif env_dict['Normalization'] == 'Normalization':
        m, s = df['mod'].describe()[['min', 'max']]
        df['mod'] = (df['mod'] - m) / (s - m)

    components = []

    components.append(
        html.Div([
            dcc.Graph(figure=px.histogram(df, x=column, marginal='box', color='Y_LABEL',
                                          barmode='overlay', nbins=20, title='Original Distribution'),
                      style={'float': 'left', 'background-color': 'rgb(230, 230, 230)', 'width': '50%'}),

            dcc.Graph(figure=px.histogram(df, x='mod', color='Y_LABEL', marginal='box',
                                          barmode='overlay', nbins=20, title='Modified Distribution'),
                      style={'float': 'left', 'background-color': 'rgb(230, 230, 230)', 'width': '50%'}),
        ], style={'float': 'left', 'display': 'flex', 'flex-direction': 'row', 'width': '1024px'})
    )

    return html.Div(components, style={'float': 'top', 'width': '1024px'})


def make_categorical_layout(df, column, mods=[]):
    components = []

    components.append(
        html.Div([
            dcc.Graph(figure=px.histogram(df, x=column, color='Y_LABEL', barmode='overlay', title='Original Distribution'),
                      style={'float': 'left', 'background-color': 'rgb(230, 230, 230)', 'width': '50%'}),

            dcc.Graph(figure=px.histogram(df, x=column, color='Y_LABEL', barmode='overlay', title='Modified Distribution'),
                      style={'float': 'left', 'background-color': 'rgb(230, 230, 230)', 'width': '50%'}),

        ], style={'float': 'left', 'display': 'flex', 'flex-direction': 'row', 'width': '1024px'})
    )
    return html.Div(components, style={'float': 'top', 'width': '1024px'})


column = df.columns[0]


@app_dash.callback(Output('layouts', 'children'),
                   [Input('column_type', 'value'),
                    Input('normalization', 'value'),
                    Input('scale', 'value'),
                    Input('missing', 'value'),
                    Input('column', 'value'),])
def update_layout(column_type, normalization, scale, missing, column):

    if column is None:
        column = df.columns[0]

    env_dict['Column Name'] = column
    env_dict['Column Type'] = column_type
    env_dict['Normalization'] = normalization
    env_dict['Scale'] = scale
    env_dict['Missing'] = missing

    return make_layout(df, column, column_type)


from save_query import save_query

@app_dash.callback(Output('query_box', 'children'),
                     [Input('query', 'n_clicks')])
def make_query(n_clicks):
    if n_clicks is not None:
        text = save_query(env_dict)
        print(text)
        return dcc.Textarea(value = text)

app_dash.layout = html.Div(children=[
    make_top(),
    html.Div([
        html.Button('Make Query', id='query'),
        html.Div(id='query_box'),
    ], style = {'width': '1024px'}),
    ], style= { 'width': '1024px'})

# Now create your regular FASTAPI application
app = FastAPI()
# Now mount you dash server into main fastapi application
app.mount("/dash", WSGIMiddleware(app_dash.server))
