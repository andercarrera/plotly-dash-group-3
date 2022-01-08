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
                    id='check_outliers',
                    options=[
                        {'label': 'Show graph without outliers', 'value': 'outlier'}
                    ],
                    value=[]
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

            #Homogeneity KMeans

            html.Div([
                html.H1("Homogeneity"),
                html.H3(id="homogeneity_kmeans_text")
            ],
                id="homogeneity-kmeans",
                className="mini_container indicator",
            ),

            # Homogeneity DBSCAN

            html.Div([
                html.H1("Homogeneity"),
                html.H3(id="homogeneity_dbscan_text")
            ],
                id="homogeneity-dbscan",
                className="mini_container indicator",
            ),

            # Completeness KMeans

            html.Div([
                html.H1("Completeness"),
                html.H3(id="completeness_kmeans_text")
            ],
                id="completeness-kmeans",
                className="mini_container indicator",
            ),

            # Completeness DBSCAN

            html.Div([
                html.H1("Completeness"),
                html.H3(id="completeness_dbscan_text")
            ],
                id="completeness-dbscan",
                className="mini_container indicator",
            ),

            # V Measure KMeans

            html.Div([
                html.H1("V-Measure"),
                html.H3(id="vmeasure_kmeans_text")
            ],
                id="vmeasure-kmeans",
                className="mini_container indicator",
            ),

            # V Measure DBSCAN

            html.Div([
                html.H1("V-Measure"),
                html.H3(id="vmeasure_dbscan_text")
            ],
                id="vmeasure-dbscan",
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
                                      title="Clustering result",
                                      labels={'Coord 1': 'Coordinate 1',
                                              'Coord 2': 'Coordinate 2'})
                ),

            ],
                id="scatter",
                style={'display': 'block'},
            ),
            html.Div([
                dcc.Graph(
                    id='scatter-graph-dbscan',
                    figure=px.scatter(dashboard.pca, x="Coord 1", y="Coord 2", color="Labels",
                                      title="Clustering result",
                                      labels={'Coord 1': 'Coordinate 1',
                                              'Coord 2': 'Coordinate 2'})
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
            html.Div([
                dcc.Graph(id='no-outlier-graph'),
            ],
                id="no-outlier",
                style={'display': 'block'},
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

        ],
            id="graphs2",
        )
    ],
        id="dashboard",
        className="flex-display",
    ),
])


@app.callback(
    Output(component_id='no-outlier', component_property='style'),
    [Input(component_id='check_outliers', component_property='value')])
def show_hide_element(value):
    if value == ['outlier']:
        return {'display': 'block'}
    else:
        return {'display': 'none'}

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
    Output(component_id='homogeneity-kmeans', component_property='style'),
    Output(component_id='homogeneity-dbscan', component_property='style'),
    Output(component_id='completeness-kmeans', component_property='style'),
    Output(component_id='completeness-dbscan', component_property='style'),
    Output(component_id='vmeasure-kmeans', component_property='style'),
    Output(component_id='vmeasure-dbscan', component_property='style'),
    Output(component_id='outliers', component_property='style'),
    [Input(component_id='algorithm-dropdown', component_property='value')])
def show_hide_element(visibility_state):
    if visibility_state == 'DBSCAN':
        return {'display': 'block'}, {'display': 'block'}, {'display': 'none'}, {'display': 'block'}, {'display': 'none'}, {'display': 'block'}, {'display': 'none'}, {'display': 'block'}, {'display': 'none'}, {'display': 'block'}, {'display': 'block'}
    else:
        return {'display': 'none'}, {'display': 'none'}, {'display': 'block'}, {'display': 'none'}, {'display': 'block'}, {'display': 'none'}, {'display': 'block'}, {'display': 'none'}, {'display': 'block'}, {'display': 'none'}, {'display': 'none'}


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
     Output("silhouette_text", "children"),
     Output("homogeneity_kmeans_text", "children"),
     Output("completeness_kmeans_text", "children"),
     Output("vmeasure_kmeans_text", "children")],
    [Input('k_slider', 'value'), Input("algorithm-dropdown", "value")])
def update_k_param(value, algorithm):
    dashboard.update_model(algorithm)
    dashboard.update_k_param(value)
    fig = px.scatter(dashboard.pca, x="Coord 1", y="Coord 2", color="Labels", title="Clustering result",
                     labels={'Coord 1': 'Coordinate 1',
                             'Coord 2': 'Coordinate 2'})
    silhouette, homogeneity, completeness, vmeasure = dashboard.get_indicators()
    return fig, silhouette, homogeneity, completeness, vmeasure


@app.callback(
    [Output('scatter-graph-dbscan', 'figure'),
     Output('silhouette_dbscan_text', 'children'),
     Output("homogeneity_dbscan_text", "children"),
     Output("completeness_dbscan_text", "children"),
     Output("vmeasure_dbscan_text", "children"),
     Output('no-outlier-graph', 'figure')],
    [Input('eps_slider', 'value'), Input('slider_min_samples', 'value'), Input("algorithm-dropdown", "value")])
def update_dbscan_params(eps, min_samples, value):
    dashboard.update_model(value)
    dashboard.update_dbscan_params(eps, min_samples)
    fig = px.scatter(dashboard.pca, x="Coord 1", y="Coord 2", color="Labels", title="Clustering result",
                     labels={'Coord 1': 'Coordinate 1',
                             'Coord 2': 'Coordinate 2'})

    silhouette, homogeneity, completeness, vmeasure = dashboard.get_indicators()

    fig2 = px.scatter(dashboard.df_no_outliers, x="Coord 1", y="Coord 2", color="Labels", title="Clustering result without outliers",
                     labels={'Coord 1': 'Coordinate 1',
                             'Coord 2': 'Coordinate 2'})

    return fig, silhouette, homogeneity, completeness, vmeasure, fig2

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
