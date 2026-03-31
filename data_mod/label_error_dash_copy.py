"""
This code was written by Léo Dechaumet from Mines Paris PSL"""


import io
import os
import base64

import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import numpy as np
from sklearn.preprocessing import StandardScaler
import umap
from dash import Dash, html, dcc, Input, Output, Patch, callback, State

from skimage.morphology import footprint_rectangle, closing, opening, reconstruction, disk
from skimage.morphology import remove_small_objects
import cv2
from skimage.filters import rank
from skimage import exposure
from skimage.measure import label, regionprops
from skimage.morphology import skeletonize
from scipy import ndimage as ndi
import matplotlib.pyplot as plt

# --- Création de la figure (comme la tienne) ---
fig = go.Figure()

path_test = "test_temp"

class_names = sorted([e.lower() for e in os.listdir(path_test)])

feature_name = np.load("morpho_features_name.npy", allow_pickle=True)


# --- App Dash ---
app = dash.Dash(__name__)

app.layout = html.Div([
    dcc.Store(id="umap-data"),

    # Ligne 1 : Radio + Select
    html.Div([

        dcc.Dropdown(
            options=[

                {"label": "fold 0", "value":0},
                {"label": "fold 1", "value":1},
                {"label": "fold 2", "value":2},
                {"label": "fold 3", "value":3},
                {"label": "fold 4", "value":4},
                #{"label": "morpho", "value":5},
            ],
            value=0,
            id="fold-dropdown",
            style={"width": "200px", "marginLeft": "20px"}
        ),
    ],
    style={
        "display": "flex",
        "alignItems": "center",
        "marginBottom": "20px"
    }),

    html.Div([
        dcc.Dropdown(
            options=[{"label": f, "value": i} for i,f in enumerate(feature_name.tolist())],
            id="feature-dropdown",
            placeholder="Select features",
            multi=True
        ),
        dcc.Button("Submit", id="features-submit", n_clicks=0)

    ], style={
        "display": "flex",
        "alignItems": "center",
        "marginBottom": "20px"
    }),

    

    # Ligne 2 : Graph + Image
    html.Div([
        dcc.Store(id="mask-state", data=False),
        dcc.Store(id="plot-state", data=False),
        dcc.Graph(
            id="umap-graph",
            #figure=fig,
            style={"flex": "2"}
        ),
        html.Div([
            html.Img(
                id="image-display",
                style={
                    "marginLeft": "20px",
                    "maxWidth": "400px",
                    "height": "auto",
                    "flex": "1"
                }
            ),
            html.Img(id="mask-display", style={"marginTop": "20px", "maxWidth": "400px", "height": "auto"})
        ],style={
    "display": "flex",
    "flexDirection": "column",
    "marginLeft": "20px"
})

    ],
    style={
        "display": "flex",
        "alignItems": "flex-start"
    })

])


    #return "path/to/default/mask.png"

# --- Callback clic ---
@app.callback(
    Output("image-display", "src"),
    Input("umap-graph", "clickData"),
    Input("mask-state", "data")
)
def display_image(clickData, mask_state):
    if clickData is None:
        return ""

    img_path = clickData["points"][0]["customdata"]

    if not os.path.exists(img_path):
        return ""

    # Encodage en base64
    encoded = base64.b64encode(open(img_path, 'rb').read()).decode()

    return f"data:image/png;base64,{encoded}"


@app.callback(
    Output("umap-data", "data"),
    Input("fold-dropdown", "value"),
)
def compute_umap(fold_value):

    i = fold_value

    features = np.load(f"out_cross_val/features_fold_{i}.npy")
    labels = np.load(f"out_cross_val/labels_fold_{i}.npy")
    #probas = np.load(f"out_cross_val/probas_fold_{i}.npy")
    fname = np.load(f"out_cross_val/fname_fold_{i}.npy", allow_pickle=True)

    X_scaled = StandardScaler().fit_transform(features)

    feat_umap = umap.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        metric="euclidean",
        n_components=2
    ).fit_transform(X_scaled)

    return {
        "umap": feat_umap.tolist(),
        "labels": labels.tolist(),
        #"probas": probas.tolist(),
        "fname": fname.tolist()
    }

@app.callback(
    Output("umap-data", "data", allow_duplicate=True),
    Input("features-submit", "n_clicks"),
    State("feature-dropdown", "value"),
    prevent_initial_call=True
)
def update_feature_dropdown(n_clicks, selected_features):
    features = np.load("morpho_features.npy")
    labels = np.load("morpho_labels.npy")
    fname = np.load("morpho_fname.npy", allow_pickle=True)

    selected_features = np.array(selected_features) if len(selected_features) > 1 else np.arange(features.shape[1])
    features = features[:, selected_features]

    if features.shape[1] > 2:
        X_scaled = StandardScaler().fit_transform(features)

        feat_umap = umap.UMAP(
            n_neighbors=15,
            min_dist=0.1,
            metric="euclidean",
            n_components=2
        ).fit_transform(X_scaled)
    else:
        feat_umap = features

    return {
        "umap": feat_umap.tolist(),
        "labels": labels.tolist(),
        #"probas": probas.tolist(),
        "fname": fname.tolist()
    }


@app.callback(
    Output("umap-graph", "figure"),
    Input("umap-data", "data"),
)
def update_graph(stored_data):

    if stored_data is None:
        return go.Figure()

    feat_umap = np.array(stored_data["umap"])
    labels = np.array(stored_data["labels"])
    #probas = np.array(stored_data["probas"])
    fname = np.array(stored_data["fname"])

    #preds = probas.argmax(axis=1)


    class_names = sorted(
        [e.lower() for e in os.listdir(path_test)]
    )

    new_fig = go.Figure()

    for class_id, class_name in enumerate(class_names):
        mask = labels == class_id
        if not np.any(mask):
            continue

        new_fig.add_trace(
            go.Scattergl(
                x=feat_umap[mask, 0],
                y=feat_umap[mask, 1],
                customdata=fname[mask],
                mode="markers",
                name=class_name,
                marker=dict(size=4),
            )
        )

    new_fig.update_layout(width=1200, height=900)

    return new_fig


if __name__ == "__main__":
    app.run(debug=True)
