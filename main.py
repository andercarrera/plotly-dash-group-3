import dash
import plotly.graph_objs as go
import plotly.express as px
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import processing

dashboard = processing.Dashboard()

# Create app layout
app = dash.Dash(__name__)
app.layout = html.Div([
    # Title
    html.Div([
        html.H1(
            "Wine Data Set",
            style={"margin-bottom": "0px"},
        ),
        html.H3(
            "Visualización Datos MACC",
            style={"margin-bottom": "0px"},
        ),
    ],
        id="title",
        className="one-half column",
    ),

    # dashboard
    html.Div([
        # filters
        html.Div([
            html.P("Filter classification algorithm:"),
            dcc.Dropdown(
                id='algorithm-dropdown',
                clearable=False,
                options=[
                    # {'label': 'Decision Tree', 'value': 'DecisionTree'},
                    # {'label': 'RandomForest Classifier', 'value': 'RandomForest'},
                    # {'label': 'KNN Classifier', 'value': 'KNN'}
                    {'label': 'KMeans', 'value': 'KMeans'},
                    {'label': 'DBSCAN', 'value': 'DBSCAN'},
                    {'label': 'MeanShift', 'value': 'MeanShift'}

                ],
                value='KMeans'  # 'DecisionTree'
            ),

            html.P("Filter instances"),
            dcc.Dropdown(
                id='instances-dropdown',
                clearable=False
            ),

            html.P("Filter correlation attributes"),
            dcc.Dropdown(
                id='correlation-dropdown',
                options=dashboard.get_variable_names(),
                multi=True,
            ),

            # Barra para modificar parámetro eps
            html.Div(id='slider_div_eps', children=[
                html.P("Select the eps value"),
                dcc.Slider(
                    id='eps_slider',
                    min=0.2,
                    max=0.7,
                    step=0.05,
                    marks={0.2: '0.2',
                           0.25: '0.25',
                           0.3: '0.3',
                           0.35: '0.35',
                           0.4: '0.4',
                           0.45: '0.45',
                           0.5: '0.5',
                           0.55: '0.55',
                           0.6: '0.6',
                           0.65: '0.65',
                           0.7: '0.7',
                           },
                    value=0.5,
                )
            ], style={'display': 'block'}),

            # KNMeans Barra
            html.Div(id='slider', children=[
                html.P("Select the K value"),
                dcc.Slider(
                    id='k_slider',
                    min=2,
                    max=9,
                    marks={i: '{}'.format(i) for i in range(2,10)},
                    value=3,
                )
            ], style={'display': 'block'}),

            # Max Depth
            html.Div(id='slider_MD', children=[
                html.P("Select the maximum depth"),
                dcc.Slider(
                    min=1,
                    max=100,
                    step=1,
                    value=20,
                    # tooltip={"placement": "bottom", "always_visible": True},
                )
            ], style={'display': 'block'}),

            # Max Iter
            # html.Div(id='slider_MI', children=[
            #     html.P("Select the maximum depth"),
            #     dcc.Slider(
            #         min=0,
            #         max=9,
            #         marks={i: '{}'.format(i) for i in range(10)},
            #         value=0,  # Default = None según SKLearn
            #     )
            # ], style={'display': 'block'})

            #Slider para elegir valor min samples (DBSCAN)
            html.Div(id='slider_div_min_samples', children=[
                html.P("Select the minimum samples in a neighbourhood"),
                dcc.Slider(
                    id='slider_min_samples',
                    min=2,
                    max=9,
                    marks={i: '{}'.format(i) for i in range(2,10)},
                    value=5,
                )
            ], style={'display': 'block'}),

            html.Div(id='outliers', children=[
                dcc.Checklist(
                    options=[
                        {'label': 'Quitar outliers', 'value': 'outlier'}
                    ],
                    value=['outlier']
                )
            ], style={'display': 'block'}),
        ],
            id="filters",
            className="container",
        ),

        # indicators
        html.Div([
            # Silhouette
            html.Div([
                html.H1("Silhouette"),
                html.H3(id="silhouette_text")
            ],
                id="silhouette",
                className="mini_container indicator",
            ),

            # Silhouette
            html.Div([
                html.H1("Silhouette"),
                html.H3(id="silhouette_dbscan_text")
            ],
                id="silhouette-dbscan",
                className="mini_container indicator",
            ),

        ],
            id="indicators",
        ),

        html.Div([
            # html.Div([
            #     dcc.Graph(id='xai-graph'),
            # ],
            #     id="xai",
            # ),

            html.Div([
                dcc.Graph(
                    id='scatter-graph',
                    figure=px.scatter(dashboard.pca, x="Coord 1", y="Coord 2", color="Labels",
                                      title="Resultado de clusterización",
                                      labels={'Coord 1': 'Coordenada 1',
                                              'Coord 2': 'Coordenada 2'})
                ),

            ],
                id="scatter",
                style={'display': 'block'},
            ),
            html.Div([
                dcc.Graph(
                    id='scatter-graph-dbscan',
                    figure=px.scatter(dashboard.pca, x="Coord 1", y="Coord 2", color="Labels",
                                      title="Resultado de clusterización",
                                      labels={'Coord 1': 'Coordenada 1',
                                              'Coord 2': 'Coordenada 2'})
                ),

            ],
                id="scatter_dbscan",
                style={'display': 'block'},
            ),
            html.Div([
                dcc.Graph(
                    id='elbow-graph',
                    figure=px.line(x=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], y=dashboard.wcss, title='Elbow method',
                                   labels={
                                       'x': 'Valores para K',
                                       'y': 'WCSS',
                                   }
                                   )
                ),
            ],
                id="elbow",
            ),

        ],
            id="graphs",
        ),
        html.Div([
            html.Div([
                dcc.Graph(id='correlation-graph'),
            ],
                id="correlation",
            ),
            html.Div([
                dcc.Graph(id='silhouette-graph'),
            ],
                id="silhouette-g",
                style={'display': 'block'},
            ),
        ],
            id="graphs2",
        )
    ],
        id="dashboard",
        className="flex-display",
    ),
])


@app.callback(
    Output(component_id='slider', component_property='style'),
    [Input(component_id='algorithm-dropdown', component_property='value')])
def show_hide_element(visibility_state):
    if visibility_state == 'KMeans':
        return {'display': 'block'}
    else:
        return {'display': 'none'}


@app.callback(
    Output(component_id='slider_div_eps', component_property='style'),
    Output(component_id='slider_div_min_samples', component_property='style'),
    Output(component_id='silhouette', component_property='style'),
    Output(component_id='silhouette-dbscan', component_property='style'),
    Output(component_id='outliers', component_property='style'),
    [Input(component_id='algorithm-dropdown', component_property='value')])
def show_hide_element(visibility_state):
    if visibility_state == 'DBSCAN':
        return {'display': 'block'}, {'display': 'block'}, {'display': 'none'}, {'display': 'block'}, {'display': 'block'}
    else:
        return {'display': 'none'}, {'display': 'none'}, {'display': 'block'}, {'display': 'none'}, {'display': 'none'}


@app.callback(
    Output(component_id='scatter', component_property='style'),
    [Input(component_id='algorithm-dropdown', component_property='value')])
def show_hide_element(visibility_state):
    if visibility_state == 'KMeans':
        return {'display': 'block'}
    else:
        return {'display': 'none'}


@app.callback(
    Output(component_id='scatter_dbscan', component_property='style'),
    [Input(component_id='algorithm-dropdown', component_property='value')])
def show_hide_element(visibility_state):
    if visibility_state == 'DBSCAN':
        return {'display': 'block'}
    else:
        return {'display': 'none'}


@app.callback(
    Output(component_id='slider_MD', component_property='style'),
    [Input(component_id='algorithm-dropdown', component_property='value')])
def show_hide_element(visibility_state):
    if visibility_state == 'RandomForest':
        return {'display': 'block'}
    else:
        return {'display': 'none'}


# @app.callback(
#     Output(component_id='slider_MI', component_property='style'),
#     [Input(component_id='algorithm-dropdown', component_property='value')])
# def show_hide_element(visibility_state):
#     if visibility_state == 'DecisionTree':
#         return {'display': 'block'}
#     else:
#         return {'display': 'none'}


@app.callback(
    Output(component_id='elbow', component_property='style'),
    [Input(component_id='algorithm-dropdown', component_property='value')])
def show_hide_element(visibility_state):
    if visibility_state == 'KMeans':
        return {'display': 'block'}
    else:
        return {'display': 'none'}


@app.callback(
    [Output('scatter-graph', 'figure'),
     Output("silhouette_text", "children")],
    [Input('k_slider', 'value')])
def update_k_param(value):
    dashboard.update_k_param(value)
    fig = px.scatter(dashboard.pca, x="Coord 1", y="Coord 2", color="Labels", title="Resultado de clusterización",
                     labels={'Coord 1': 'Coordenada 1',
                             'Coord 2': 'Coordenada 2'})
    silhouette = dashboard.get_indicators()

    return fig, silhouette


@app.callback(
   [ Output('scatter-graph-dbscan', 'figure'),
     Output('silhouette_dbscan_text', 'children')],
    [Input('eps_slider', 'value'), Input('slider_min_samples', 'value')])
def update_dbscan_params(eps, min_samples):
    dashboard.update_dbscan_params(eps, min_samples)
    fig = px.scatter(dashboard.pca, x="Coord 1", y="Coord 2", color="Labels", title="Resultado de clusterización",
                     labels={'Coord 1': 'Coordenada 1',
                             'Coord 2': 'Coordenada 2'})

    silhouette = dashboard.get_indicators()

    return fig, silhouette


@app.callback(
    [
        # Output("accuracy_text", "children"),
        # Output("precision_text", "children"),
        # Output("recall_text", "children"),
        Output("instances-dropdown", "options"),
        Output("instances-dropdown", "value")
    ],
    [Input("algorithm-dropdown", "value")],
)
def algorithm_updated(value):
    dashboard.update_model(value)
    # accuracy, precision, recall = dashboard.get_indicators()
    instances, value = dashboard.get_instances()

    return instances, value  # accuracy, precision, recall, instances, value


# @app.callback(
#     Output("xai-graph", "figure"),
#     [Input("instances-dropdown", "value")],
# )
# def instance_updated(value):
#     shap_values = dashboard.get_shap_values(value)
#     sorted_shap = {}
#     for val in sorted(shap_values, key=lambda k: abs(shap_values[k])):
#         sorted_shap[val] = shap_values[val]
#     trace = go.Bar(x=list(sorted_shap.values()), y=list(sorted_shap.keys()), orientation='h')
#     graph = {
#         'data': [trace],
#         'layout': go.Layout(
#             title='Instance Explainability',
#             xaxis={
#                 'title': 'Shapley Values',
#                 'range': [-0.75, 0.75]
#             }
#         )
#     }
#     return graph


@app.callback(
    Output("correlation-graph", "figure"),
    [Input("correlation-dropdown", "value")],
)
def correlation_updated(value):
    print(value)
    graph = {}
    if value is not None and len(value) == 2:
        cols = dashboard.get_columns(value)
        trace = go.Scatter(x=cols.iloc[:, 0], y=cols.iloc[:, 1], mode='markers')
        graph = {
            'data': [trace],
            'layout': go.Layout(
                title='Correlation',
                xaxis={
                    'title': value[0],
                },
                yaxis={
                    'title': value[1],
                }
            ),
        }
    return graph


if __name__ == '__main__':
    app.run_server(debug=True)
