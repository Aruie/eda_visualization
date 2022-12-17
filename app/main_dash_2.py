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

from save_query import save_query


BASE_PATH = Path(__file__).resolve().parent
TEMPLATES = Jinja2Templates(directory=str(BASE_PATH / "templates"))


# 지정 필요
# file_name = './static/data/titanic.csv'
# label_column = 'Survived'
file_name = './static/data/train.csv'
label_column = 'Y_LABEL'

# df = pd.read_csv(file_name)
df = pd.read_csv(file_name, nrows=1000)
column = df.columns[0]



env_dict = {
    'Column Name': column,
    'Column Type': 'Numerical',
    'Normalization': 'None',
    'Scale': 'None',
    'Missing': 'None',
    'Lambda Exp': '',
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
    
    def make_radio(name, items, id, disabled = []):
        tmp = []
        tmp.append(html.Div(name, style={'width': '20%', 'text-align': 'center', 'vertical-align': 'middle'}))

        if env_dict[name] in disabled:
            env_dict[name] = items[0]

        env_dict[name] = env_dict[name] if env_dict[name] not in disabled else 0
        options = [{'label' : x, 'value':x,'disabled': x in disabled }  for x in items ]
        tmp.append(dcc.RadioItems(options = options, value=env_dict[name] , id=id, style={'vertical-align': 'middle'}, ))

        return html.Div(tmp, style={'float': 'top', 'width': '100%', 'height': '20%',
                                    'display': 'flex', 'flex-direction': 'row', 'border': '1px solid black'})

    if env_dict['Column Type'] == 'Categorical':
        disabled = ['Standardization', 'Normalization', 'Log Scale', 'Exponential', 'Fill Mean','Fill Zero','Fill Median']
    if env_dict['Column Type'] == 'Numerical':
        disabled = ['Fill Mode']


    divs = []
    divs.append(make_radio('Column Type', ['Numerical', 'Categorical'], id='column_type'))

    divs.append(make_radio(
        'Missing', ['None', 'Fill Zero', 'Fill Mean', 'Fill Median','Fill Mode'], id='missing', disabled=disabled))

    divs.append(make_radio('Normalization', [
                'None', 'Standardization', 'Normalization'],  id='normalization', disabled=disabled))

    divs.append(make_radio('Scale', ['None', 'Log Scale', 'Exponential'],  id='scale', disabled=disabled))

    

    tmp = []
    tmp.append(html.Div('Lambda x', style={'width': '20%', 'text-align': 'center', 'vertical-align': 'middle'}))
    tmp.append(dcc.Textarea(id='lambda_exp', rows=1,
               style={'vertical-align': 'middle', 'width': '80%'}))
    divs.append(html.Div(tmp, style={'float': 'top', 'width': '100%', 'height': '20%',
                                     'display': 'flex', 'flex-direction': 'row', 'border': '1px solid black', 'background-color': 'rgb(250, 250, 150)'}))

    return html.Div(divs, style={'width': '70%', 'height': '150px', 'float': 'right',
                                 'display': 'flex', 'flex-direction': 'column', 'background-color': 'rgb(250, 250, 150)'})

def make_layout(df, column, col_type = 'Numerical'):
    div1 = []
    div1.append(make_column_information(df, column))
    div1.append(make_components())
    
    div1 = html.Div(div1, style={'float': 'top', 'width': '100%', 'background-color': 'rgb(250, 250, 150)'})
    div = make_graph_layout(df, column, env_dict)

    text = save_query(env_dict)
    footer = html.Div([
        dcc.Textarea(value = text, style={'width': '100%', 'height': '200px', 'background-color': 'rgb(250, 250, 150)'}, id='query')]
        )

    return html.Div([div1, div, footer], style={'float': 'top', 'width': '100%'})

def make_column_information(df, column):
    df['mod'] = df[column]

    na_cnt = df[column].isna().sum()
    length = len(df[column])
    cardinality = len(df[column].unique())

    component = dcc.Markdown(f"""
        ### Columns : {column}
        - Length : {length} 
        - NA count : {na_cnt} ({na_cnt / length * 100:.2f}%)
        - Cardinality : {cardinality} ({cardinality / length * 100:.2f}%)
        """,
                             style={'float': 'left', 'background-color': 'rgb(250, 250, 150)', 'width': '30%', 'height': '150px'})
    return component


def make_graph_layout(df, column, env_dict):

    df['mod'] = df[column]

    try : 
        if env_dict['Missing'] == 'Fill Zero':
            df['mod'] = df['mod'].fillna(0)
        elif env_dict['Missing'] == 'Fill Mean' :
            df['mod'] = df['mod'].fillna(df['mod'].mean())
        elif env_dict['Missing'] == 'Fill Median' :
            df['mod'] = df['mod'].fillna(df['mod'].median())
        elif env_dict['Missing'] == 'Fill Mode' :
            df['mod'] = df['mod'].fillna(df['mod'].mode()[0])

        if env_dict['Scale'] == 'Log Scale' :
            df['mod'] = np.log(df['mod'] + 1)
        elif env_dict['Scale'] == 'Exponential' :
            df['mod'] = np.exp(df['mod'])

        if env_dict['Normalization'] == 'Standardization':
            m, s = df['mod'].describe()[['mean', 'std']]
            df['mod'] = (df['mod'] - m) / s
        elif env_dict['Normalization'] == 'Normalization':
            m, s = df['mod'].describe()[['min', 'max']]
            df['mod'] = (df['mod'] - m) / (s - m)

        if env_dict['Lambda Exp']:
            print(env_dict['Lambda Exp'])
            df['mod'] = df['mod'].apply(lambda x : eval(env_dict['Lambda Exp']))
        
    except Exception as e: 
        print(f'error occured when modifying data : {e}') 
        graph = False
    else : 
        graph = True

    if graph == True:

        components = []

        if env_dict['Column Type'] == 'Numerical':
            components.append(html.Div([
                dcc.Graph(figure=px.histogram(df, x=column, marginal='box', color=label_column,
                                            barmode='overlay', nbins=20, title='Original Distribution'),
                        style={'float': 'left', 'background-color': 'rgb(230, 230, 230)', 'width': '50%'}),

                dcc.Graph(figure=px.histogram(df, x='mod', color=label_column, marginal='box',
                                            barmode='overlay', nbins=20, title='Modified Distribution'),
                        style={'float': 'left', 'background-color': 'rgb(230, 230, 230)', 'width': '50%'}),
            ], style={'float': 'left', 'display': 'flex', 'flex-direction': 'row', 'width': '1024px'})
        )

        elif env_dict['Column Type'] == 'Categorical':
            components.append(html.Div([
                dcc.Graph(figure=px.histogram(df, x=column, color=label_column, barmode='overlay', title='Original Distribution'),
                        style={'float': 'left', 'background-color': 'rgb(230, 230, 230)', 'width': '50%'}),

                dcc.Graph(figure=px.histogram(df, x='mod', color=label_column, barmode='overlay', title='Modified Distribution'),
                        style={'float': 'left', 'background-color': 'rgb(230, 230, 230)', 'width': '50%'}),

            ], style={'float': 'left', 'display': 'flex', 'flex-direction': 'row', 'width': '1024px'})
        )

        return html.Div(components, style={'float': 'top', 'width': '1024px'})
    
    else :
        return None








@app_dash.callback(Output('layouts', 'children'),
                   [Input('column_type', 'value'),
                    Input('normalization', 'value'),
                    Input('scale', 'value'),
                    Input('missing', 'value'),
                    Input('column', 'value'),
                    Input('lambda_exp', 'value')])
def update_layout(column_type, normalization, scale, missing, column, lambda_exp):

    if column is None:
        column = df.columns[0]
    if column_type is None :
        column_type = 'Numerical'

    env_dict['Column Name'] = column
    env_dict['Column Type'] = column_type
    env_dict['Normalization'] = normalization
    env_dict['Scale'] = scale
    env_dict['Missing'] = missing
    env_dict['Lambda Exp'] = lambda_exp

    return make_layout(df, column, column_type)



app_dash.layout = html.Div(children=[make_top()
        ], style = {'width': '1024px'})


# Now create your regular FASTAPI application
app = FastAPI()
# Now mount you dash server into main fastapi application
app.mount("/dash", WSGIMiddleware(app_dash.server))
