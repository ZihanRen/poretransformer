#%%
import numpy as np
import os
import pickle
import argparse
import dash
from dash import html, dcc, Input, Output
import plotly.graph_objects as go

def load_data(ct_idx, vol_dim,root_dir,sample_idx,emsemble_idx=0):
    file_path = f'{root_dir}/sample_{ct_idx}/img_gen_vol_{vol_dim}.pkl'
    with open(file_path, 'rb') as file:
        img_results = pickle.load(file)
    
    img = img_results[sample_idx]['generate'][emsemble_idx]
    return img


# %%
def create_dash_app(data):
    app = dash.Dash(__name__)

    app.layout = html.Div([
        dcc.Graph(id='plot'),
        html.P("Slice Index:"),
        dcc.Slider(
            id='slice-index',
            min=0,
            max=data.shape[2] - 1,
            value=5,
            marks={0: '0', data.shape[2] - 1: str(data.shape[2] - 1)},
            step=1
        ),
        html.P("Axis:"),
        dcc.RadioItems(
            id='axis',
            options=[{'label': i, 'value': i} for i in ['x', 'y', 'z']],
            value='z',
            labelStyle={'display': 'inline-block'}
        )
    ])

    @app.callback(
        Output('plot', 'figure'),
        [Input('slice-index', 'value'),
         Input('axis', 'value')]
    )
    def update_figure(slice_index, axis):
        indices = {'x': 0, 'y': 1, 'z': 2}
        selected_axis = indices[axis]
        if selected_axis == 0:
            slice_data = data[slice_index, :, :]
        elif selected_axis == 1:
            slice_data = data[:, slice_index, :]
        else:
            slice_data = data[:, :, slice_index]

        fig = go.Figure(data=go.Heatmap(
            z=slice_data,
            colorscale='gray'
        ))
        fig.update_layout(
            title=f'Slice along {axis.upper()} at index {slice_index}',
            autosize=False,
            width=800,
            height=800,
            margin=dict(l=50, r=50, b=100, t=100),
        )
        return fig

    return app

# Run the app
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Load and display a 3D volume")
    parser.add_argument("ct_idx", type=int, help="Index of the CT scan")
    parser.add_argument("vol_dim", type=int, help="Dimension size of the volume")
    parser.add_argument("root_dir", type=str, default='data_ref_hard', nargs='?', help="Root directory where data is stored")
    parser.add_argument("sample_idx", type=int, default=0, nargs='?', help="Index of the sample")
    parser.add_argument("emsemble_idx", type=int, default=0, nargs='?', help="Index of the ensemble")
    args = parser.parse_args()

    data = load_data(
        args.sample_idx,
        args.vol_dim,
        args.root_dir,
        args.ct_idx,
        args.emsemble_idx
        )
    app = create_dash_app(data)
    app.run_server(debug=True)
# %%
