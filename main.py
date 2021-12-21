import dash
import plotly.graph_objs as go
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
                    {'label': 'Decision Tree', 'value': 'DecisionTree'},
                    {'label': 'RandomForest Classifier', 'value': 'RandomForest'},
                    {'label': 'KNN Classifier', 'value': 'KNN'}
                ],
                value='DecisionTree'
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

            # KNN Barra
            html.Div(id='slider', children=[
                html.P("Select the K value"),
                dcc.Slider(
                    min=0,
                    max=9,
                    marks={i: '{}'.format(i) for i in range(10)},
                    value=5,
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
                    tooltip={"placement": "bottom", "always_visible": True},
                )
            ], style={'display': 'block'}),

            # Max Iter
            html.Div(id='slider_MI', children=[
                html.P("Select the maximum depth"),
                dcc.Slider(
                    min=0,
                    max=9,
                    marks={i: '{}'.format(i) for i in range(10)},
                    value=0,  # Default = None según SKLearn
                )
            ], style={'display': 'block'})
        ],
            id="filters",
            className="container",
        ),

        # indicators
        html.Div([
            # Precision
            html.Div([
                html.H1("Accuracy"),
                html.H3(id="accuracy_text")
            ],
                id="accuracy",
                className="mini_container indicator",
            ),

            # Recall
            html.Div([
                html.H1("F1-Score"),
                html.H3(id="f1_text")
            ],
                id="f1-score",
                className="mini_container indicator",
            ),

            # F1-Score
            html.Div([
                html.H1("ROC AUC"),
                html.H3(id="rocauc_text")
            ],
                id="roc-auc",
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
                dcc.Graph(id='correlation-graph'),
            ],
                id="correlation",
            )
        ],
            id="graphs",
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
    if visibility_state == 'KNN':
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
    Output(component_id='slider_MI', component_property='style'),
    [Input(component_id='algorithm-dropdown', component_property='value')])
def show_hide_element(visibility_state):
    if visibility_state == 'DecisionTree':
        return {'display': 'block'}
    else:
        return {'display': 'none'}


@app.callback(
    [
        Output("accuracy_text", "children"),
        Output("f1_text", "children"),
        #Output("rocauc_text", "children"),
        Output("instances-dropdown", "options"),
        Output("instances-dropdown", "value")
    ],
    [Input("algorithm-dropdown", "value")],
)
def algorithm_updated(value):
    dashboard.update_model(value)
    accuracy, f1 = dashboard.get_indicators()
    instances, value = dashboard.get_instances()
    return accuracy, f1, instances, value


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