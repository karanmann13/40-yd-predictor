import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go

# read data from csv files
data1 = pd.read_csv("combinedata2023.csv")
data2 = pd.read_csv("combinedata2022.csv")
data3 = pd.read_csv("combinedata2021.csv")
data4 = pd.read_csv("combinedata2020.csv")
data5 = pd.read_csv("combinedata2019.csv")
data6 = pd.read_csv("combinedata2018.csv")
data7 = pd.read_csv("combinedata2017.csv")
data8 = pd.read_csv("combinedata2016.csv")
data9 = pd.read_csv("combinedata2015.csv")

# combine data 
combinedata = pd.concat([data1,data2,data3,data4,data5,data6,data7,data8,data9])

# removing unnecessary features 
columns_to_keep = ["Vertical", "Wt", "40yd", "Broad Jump"]
combinedata = combinedata[columns_to_keep]

# removing observations with null values
combinedata = combinedata.dropna()

# separate features and response
X = combinedata[["Wt", "Vertical", "Broad Jump"]]
y = combinedata["40yd"]

# develop data pipeline 
model = make_pipeline(
    StandardScaler(),
    PolynomialFeatures(degree=2, include_bias=False),
    LinearRegression()
)

# fit model
model.fit(X,y)

# define function to predict new observations
def predict_output(new_dt):
    return model.predict(new_dt)

# initialize app 
app = dash.Dash(__name__)

app.layout = html.Div([
    
    # header
    html.H1("40 yd dash predictor"),

    # develop interface for inputting values
    html.Div([
        html.Label("Weight"),
        dcc.Input(id='wt-input', type='number', value = 0),
    ]),

    html.Div([
        html.Label("Vertical Jump"),
        dcc.Input(id="vert-input", type='number', value = 0),
    ]),

    html.Div([
        html.Label("Broad Jump"),
        dcc.Input(id="broad-input", type='number', value = 0),
    ]),

    html.Button('Predict', id="predict-button"),

    html.Div(id='prediction-output')
])

@app.callback(
    [Output('prediction-output', 'children')],
    [Input('predict-button', 'n_clicks')],
    [State('wt-input', 'value'),
     State('vert-input', 'value'),
     State('broad-input', 'value')]
)

def update_output(n_clicks, x1, x2, x3):

    if n_clicks is None:
        return ''
    
    new_data = pd.DataFrame({'Wt': [x1], 'Vertical': [x2], 'Broad Jump': [x3]})
    prediction = predict_output(new_data)

    return [f'The predicted value is: {prediction[0]}']

if __name__ == '__main__':
    app.run_server(debug=True)