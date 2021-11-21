import numpy as np
from random import random
from matplotlib import pyplot as plt

from sections_cube import get_n_plates

import dash
import dash_core_components as dcc
import dash_html_components as html

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div(children=[
       html.Div([
           dcc.Graph(id='fig1'),
       ]) ,
       html.Div([
           html.H6('Длина стороны'),
           dcc.Slider(
               id='slider-side',
               min=0,
               max=10,
               step=1,
               value=0.5,
               marks={i: str(i) for i in np.linspace(0, 10, 100)}
           )
#            html.H6('Количество пивоварен в топе'),
#            dcc.Slider(
#                id='slider-top1',
#                min=0,
#                max=500,
#                step=50,
#                value=500,
#                marks={i: str(i) for i in range(0, 500, 50)})
       ])
])

@app.callback(
   dash.dependencies.Output('fig1', 'figure'),
   [dash.dependencies.Input('slider-side', 'value')])
def output_fig(side):
    fig = plt.plot(generates_n_plates(1000, side=side))
    return fig

if __name__ == '__main__':
   app.run_server(debug=True)