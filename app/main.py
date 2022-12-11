import pandas as pd
import os
from fastapi import FastAPI, APIRouter, Query, HTTPException, Request
from pydantic import BaseModel

from fastapi.templating import Jinja2Templates

from typing import Optional, Any
from pathlib import Path

from fastapi.staticfiles import StaticFiles


#
BASE_PATH = Path(__file__).resolve().parent
TEMPLATES = Jinja2Templates(directory=str(BASE_PATH / "templates"))


class FileInfo(BaseModel):
    name: str
    columns: list
    size: int


# 1
app = FastAPI(
    title="Recipe API", openapi_url="/openapi.json"
)
app.mount("/static", StaticFiles(directory="static"), name="static")

api_router = APIRouter()


#####################


def get_file_list():
    files = []
    for file in Path("./static/data").glob("*.csv"):
        tmp = {}
        tmp['name'] = file.name

        try:
            tmp['columns'] = list(pd.read_csv(file, nrows=0))
        except:
            tmp['columns'] = list(pd.read_csv(file, nrows=0, encoding='cp949'))

        tmp['size'] = os.path.getsize(file)
        files.append(tmp)
    return files


@api_router.get("/", status_code=200)
def root(request: Request) -> Any:
    """
    Root GET
    """
    file_list = get_file_list()
    print(file_list)

    return TEMPLATES.TemplateResponse(
        "index.html",
        {"request": request, 'file_list': file_list},
    )

@api_router.get("/eda", status_code=200)
def root(request: Request) -> Any:
    """
    Root GET
    """
    file_name = './static/data/iris.csv'
    df = pd.read_csv(file_name)

    columns = list(df)

    print(columns)

    return TEMPLATES.TemplateResponse(
        "eda.html",
        {'request': request,'df': df
         },
    )


@api_router.get("/eda/{column}", status_code=200)
def anly_column(request: Request, column: str) -> Any:
    file_name = './static/data/iris.csv'
    df = pd.read_csv(file_name)

    df_sub = df[[column, 'variety']]
    
    import matplotlib.pyplot as plt
    import seaborn as sns

    
    sns.boxplot(x='variety', y=column, data=df_sub)
    
    plt.show()


    return TEMPLATES.TemplateResponse(
        "eda.html",
        {'request': request,'df': df_sub, 'column': column,
            'image' : 'static/images/tmp.png'
         },
    )

app.include_router(api_router)
# 5
if __name__ == "__main__":
    # Use this for debugging purposes only
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="debug")
