#%%
import pandas as pd

data = pd.read_csv('app/static/data/iris.csv')
# %%
for k, v  in data['sepal.length'].describe().items():
    print(k,v)
    break
# %%
