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


file_name = './static/data/iris.csv'
df = pd.read_csv(file_name)

# Create the Dash application, make sure to adjust requests_pathname_prefx
app_dash = Dash(__name__, requests_pathname_prefix='/dash/')










def make_layout(df, column):

    df = df.copy()
    df[column].describe().to_dict()
    
    
    df['logscaled'] = np.log(df[column] + 1)

    m, s = df[column].describe()[['mean', 'std']]
    df['normalized'] = (df[column] - m) / s



    na_cnt = df[column].isna().sum()
    length = len(df[column])
    cardinality = len(df[column].unique())

    components = []

    components.append(
        dcc.Markdown(f"""
        # {file_name.split('/')[-1]} : {column}
        ## Summary
        - Length : {length} 
        - NA count : {na_cnt} ({na_cnt / length * 100:.2f}%)
        - Cardinality : {cardinality} ({cardinality / length * 100:.2f}%)

        ## Distribution
        """, style={'background-color': 'rgb(250, 250, 150)'})
        )

    components.append(
        dcc.Graph(figure = px.histogram(df, x=column, marginal='box', color='variety', 
            barmode='overlay', nbins=20, title='Original Distribution'), 
            style={'float': 'left', 'background-color': 'rgb(230, 230, 230)'})
                ), 
    
    components.append(
        dcc.Graph(figure = px.histogram(df, x='normalized', color='variety', marginal='box', 
            barmode='overlay', nbins=20, title='Normalized Distribution'), 
            style={'float': 'left', 'background-color': 'rgb(230, 230, 230)'}))

    components.append(
        dcc.Graph(figure = px.histogram(df, x='logscaled', color='variety', marginal='box', 
            barmode='overlay', nbins=20, title='Normalized Distribution'), 
            style={'float': 'left', 'background-color': 'rgb(230, 230, 230)'}))

    return html.Div(components)




column = df.columns[0]
app_dash.layout = make_layout(df, column)


# Now create your regular FASTAPI application
app=FastAPI()

# Now mount you dash server into main fastapi application
app.mount("/dash", WSGIMiddleware(app_dash.server))

@ app.get("/hello_fastapi")
def read_main():
    return {"message": "Hello World"}

@ app.get("/eda/{column}", status_code = 200)
def anly_column(request: Request, column: str) -> Any:
    file_name='./static/data/iris.csv'
    df=pd.read_csv(file_name)

    df_sub=df[[column, 'variety']]

    return TEMPLATES.TemplateResponse(
        "eda.html",
        {
            'request': request, 'df': df_sub, 'column': column
        },
    )
