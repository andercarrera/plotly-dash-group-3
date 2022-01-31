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
            "Packetbeat Data Set",
            style={"margin-bottom": "0px"},
        ),
        html.H3(
            "POPBL MACC",
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
                    {'label': 'KMeans', 'value': 'KMeans'},
                    {'label': 'DBSCAN', 'value': 'DBSCAN'}
                ],
                value='KMeans'
            ),

            html.P("Filter correlation attributes"),
            dcc.Dropdown(
                id='correlation-dropdown',
                options=dashboard.get_variable_names(),
                multi=True,
            ),

            html.Div(id='score_kmeans', children=[
                html.P("Choose diagram"),
                dcc.Dropdown(
                    id='diagram-dropdown',
                    options=[
                        {'label': 'Elbow method', 'value': 'elbow'},
                        {'label': 'Silhouette score', 'value': 'silhouette'},
                        {'label': 'Calinski-Harabasz score', 'value': 'calinski'},
                        {'label': 'Davies-Bouldin score', 'value': 'davies'}
                    ],
                    value='elbow'
                )
            ]),

            # Barra para modificar parámetro eps
            html.Div(id='slider_div_eps', children=[
                html.P("Select the eps value"),
                dcc.Slider(
                    id='eps_slider',
                    min=0.01,
                    max=0.12,
                    step=0.01,
                    marks={
                        0.01: '0.01',
                        0.02: '0.02',
                        0.03: '0.03',
                        0.04: '0.04',
                        0.05: '0.05',
                        0.06: '0.06',
                        0.07: '0.07',
                        0.08: '0.08',
                        0.09: '0.09',
                        0.1: '0.1',
                        0.11: '0.11',
                        0.12: '0.12',
                           },
                    value=0.05,
                )
            ], style={'display': 'block'}),
            # KMeans Barra
            html.Div(id='slider', children=[
                html.P("Select the K value"),
                dcc.Slider(
                    id='k_slider',
                    min=2,
                    max=9,
                    marks={i: '{}'.format(i) for i in range(2, 10)},
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
                )
            ], style={'display': 'block'}),

            # Slider para elegir valor min samples (DBSCAN)
            html.Div(id='slider_div_min_samples', children=[
                html.P("Select the minimum samples in a neighbourhood"),
                dcc.Slider(
                    id='slider_min_samples',
                    min=3,
                    max=18,
                    marks={i: '{}'.format(i) for i in range(3, 18)},
                    value=8,
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
            # Silhouette KMeans
            html.Div([
                html.H2("Silhouette"),
                html.H3(id="silhouette_text")
            ],
                id="silhouette",
                className="mini_container indicator",
            ),

            # Silhouette DBSCAN
            html.Div([
                html.H2("Silhouette"),
                html.H3(id="silhouette_dbscan_text")
            ],
                id="silhouette-dbscan",
                className="mini_container indicator",
            ),

            # Calinski-Harabasz KMeans

            html.Div([
                html.H2("Calinski-Harabasz"),
                html.H3(id="calinski_kmeans_text")
            ],
                id="calinski-kmeans",
                className="mini_container indicator",
            ),

            # Calinski-Harabasz DBSCAN

            html.Div([
                html.H2("Calinski-Harabasz"),
                html.H3(id="calinski_dbscan_text")
            ],
                id="calinski-dbscan",
                className="mini_container indicator",
            ),

            # # Davies-Bouldin KMeans
            html.Div([
                html.H2("Davies-Bouldin"),
                html.H3(id="davies_kmeans_text")
            ],
                id="davies-kmeans",
                className="mini_container indicator",
            ),

            # # Davies-Bouldin DBSCAN
            html.Div([
                html.H2("Davies-Bouldin"),
                html.H3(id="davies_dbscan_text")
            ],
                id="davies-dbscan",
                className="mini_container indicator",
            ),

        ],
            id="indicators",
        ),

        html.Div([
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
                    figure=px.line(x=[2, 3, 4, 5, 6, 7, 8, 9, 10], y=dashboard.wcss, title='Elbow method',
                                   labels={
                                       'x': 'K values',
                                       'y': 'WCSS',
                                   }
                                   )
                ),
            ],
                id="elbow_graph_div",
            ),
            html.Div([
                dcc.Graph(
                    id='silhouette-graph',
                    figure=px.line(x=[2, 3, 4, 5, 6, 7, 8, 9, 10], y=dashboard.silhouette, title='Silhouette score',
                                   labels={
                                       'x': 'K values',
                                       'y': 'Silhouette value',
                                   }
                                   )
                ),
            ],
                id="silhouette_graph_div",
            ),
            html.Div([
                dcc.Graph(
                    id='calinski-graph',
                    figure=px.line(x=[2, 3, 4, 5, 6, 7, 8, 9, 10], y=dashboard.calinski,
                                   title='Calinski-Harabasz score',
                                   labels={
                                       'x': 'K values',
                                       'y': 'Calinski-Harabasz value',
                                   }
                                   )
                ),
            ],
                id="calinski_graph_div",
            ),
            html.Div([
                dcc.Graph(
                    id='davies-graph',
                    figure=px.line(x=[2, 3, 4, 5, 6, 7, 8, 9, 10], y=dashboard.davies, title='Davies-Bouldin score',
                                   labels={
                                       'x': 'K values',
                                       'y': 'Davies-Bouldin value',
                                   }
                                   )
                ),
            ],
                id="davies_graph_div",
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
    [Input(component_id='check_outliers', component_property='value'),
     Input(component_id='algorithm-dropdown', component_property='value')])
def show_hide_element(value, algorithm):
    if algorithm == 'DBSCAN':
        if value == ['outlier']:
            return {'display': 'block'}
        else:
            return {'display': 'none'}
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
    Output(component_id='calinski-kmeans', component_property='style'),
    Output(component_id='calinski-dbscan', component_property='style'),
    Output(component_id='davies-kmeans', component_property='style'),
    Output(component_id='davies-dbscan', component_property='style'),
    Output(component_id='outliers', component_property='style'),
    [Input(component_id='algorithm-dropdown', component_property='value')])
def show_hide_element(visibility_state):
    if visibility_state == 'DBSCAN':
        return {'display': 'block'}, {'display': 'block'}, {'display': 'none'}, {'display': 'block'}, {'display': 'none'}, {'display': 'block'}, {'display': 'none'}, {
                   'display': 'block'}, {'display': 'block'}
    else:
        return {'display': 'none'}, {'display': 'none'}, {'display': 'block'}, {'display': 'none'}, {'display': 'block'}, {'display': 'none'}, {'display': 'block'}, {
                   'display': 'none'}, {'display': 'none'}


@app.callback(
    Output(component_id='scatter', component_property='style'),
    [Input(component_id='algorithm-dropdown', component_property='value')])
def show_hide_element(visibility_state):
    if visibility_state == 'KMeans':
        return {'display': 'block'}
    else:
        return {'display': 'none'}


@app.callback(
    Output(component_id='score_kmeans', component_property='style'),
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


@app.callback(
    Output(component_id='elbow_graph_div', component_property='style'),
    Output(component_id='silhouette_graph_div', component_property='style'),
    Output(component_id='calinski_graph_div', component_property='style'),
    Output(component_id='davies_graph_div', component_property='style'),
    [Input(component_id='algorithm-dropdown', component_property='value'),
     Input(component_id='diagram-dropdown', component_property='value')])
def show_hide_element(algorithm, diagram):
    if algorithm == 'KMeans':
        if diagram == 'elbow':
            return {'display': 'block'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}
        if diagram == 'silhouette':
            return {'display': 'none'}, {'display': 'block'}, {'display': 'none'}, {'display': 'none'}
        if diagram == 'calinski':
            return {'display': 'none'}, {'display': 'none'}, {'display': 'block'}, {'display': 'none'}
        if diagram == 'davies':
            return {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'block'}
    else:
        return {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}


@app.callback(
    [Output('scatter-graph', 'figure'),
     Output("silhouette_text", "children"),
     Output("calinski_kmeans_text", "children"),
     Output("davies_kmeans_text", "children")],
    [Input('k_slider', 'value'), Input("algorithm-dropdown", "value")])
def update_k_param(value, algorithm):
    dashboard.update_model(algorithm)
    dashboard.update_k_param(value)
    fig = px.scatter(dashboard.pca, x="Coord 1", y="Coord 2", color="Labels", title="Clustering result",
                     labels={'Coord 1': 'Coordinate 1',
                             'Coord 2': 'Coordinate 2'})
    silhouette, calinski, davies = dashboard.get_indicators()
    return fig, silhouette, calinski, davies


@app.callback(
    [Output('scatter-graph-dbscan', 'figure'),
     Output('silhouette_dbscan_text', 'children'),
     Output("calinski_dbscan_text", "children"),
     Output("davies_dbscan_text", "children"),
     Output('no-outlier-graph', 'figure')],
    [Input('eps_slider', 'value'), Input('slider_min_samples', 'value'), Input("algorithm-dropdown", "value")])
def update_dbscan_params(eps, min_samples, value):
    dashboard.update_model(value)
    dashboard.update_dbscan_params(eps, min_samples)
    fig = px.scatter(dashboard.pca, x="Coord 1", y="Coord 2", color="Labels", title="Clustering result",
                     labels={'Coord 1': 'Coordinate 1',
                             'Coord 2': 'Coordinate 2'})

    silhouette, calinski, davies = dashboard.get_indicators()

    fig2 = px.scatter(dashboard.df_no_outliers, x="Coord 1", y="Coord 2", color="Labels", title="Clustering result without outliers",
                     labels={'Coord 1': 'Coordinate 1',
                             'Coord 2': 'Coordinate 2'})
    print(str(eps) + " " + str(min_samples) + " " + str(silhouette) + " " + str(calinski), flush=True)
    return fig, silhouette, calinski, davies, fig2


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
