import argparse
import os

from dash import Dash, html, dcc, callback, Output, Input, ctx
import pandas as pd
import plotly.graph_objects as go


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment_path",
        type=str,
        required=True,
        help="Specifies path to the experiment directory.",
    )

    return parser.parse_args()


args = parse_args()


df = pd.read_csv(os.path.join(args.experiment_path, "aggregated_df.csv"))
baselines_df = pd.read_csv("./baselines_df.csv")

app = Dash()

# Define layout
app.layout = [
    html.H1(children="Super-cool dashboard!", style={"textAlign": "center"}),
    html.H3(
        f"Experiment: {args.experiment_path}",
        style={"textAlign": "center"},
    ),
    html.P("x axis/further filter:", style={"font-weight": "bold"}),
    dcc.RadioItems(
        id="column-radios",
        options=[
            {"label": "sample_size/splits", "value": 1},
            {"label": "splits/sample_size", "value": 2},
        ],
        value=1,
        labelStyle={"display": "block"},
    ),
    html.P("further filter value:", style={"font-weight": "bold"}),
    dcc.Checklist(
        id="further-checkbox",
        options=["Show baselines", "Show all"],
        value=[],
        labelStyle={"width": "50%", "display": "block"},
    ),
    dcc.Dropdown(
        id="further-value-dd",
        options=df.splits.unique(),
        value=df.splits.unique()[0],
    ),
    html.P("series to display:", style={"font-weight": "bold"}),
    html.Button("Clear All", id="clear-button"),  # Clear all button
    html.Button("All benchmarks", id="benchmarks-button"),  # Set all benchmarks button
    dcc.Checklist(
        id="column-checkboxes",
        options=df.columns[:20],
        value=[],
        labelStyle={"width": "33%", "display": "inline-block"},
    ),
    dcc.Graph(id="graph-content"),
]


@app.callback(
    Output("further-value-dd", "disabled"),
    Input("further-checkbox", "value"),
)
def use_further_checkbox(value):
    if "Show all" in value:
        return True
    return False


@app.callback(
    Output("column-checkboxes", "value"),
    [
        Input("clear-button", "n_clicks"),
        Input("benchmarks-button", "n_clicks"),
    ],
    prevent_initial_call=True,  # Prevent triggering on initial load
)
def clear_checklist(but1, but2):
    button_id = ctx.triggered_id if not None else "No clicks yet"
    # Return an empty list to clear all selected checkboxes
    if button_id == "clear-button":
        return []
    elif button_id == "benchmarks-button":
        return df.columns[:12]


@app.callback(
    Output("further-value-dd", "options"),
    Input("column-radios", "value"),
)
def clear_checklist(value):
    if value == 1:
        return df["splits"].unique()
    return df["sample_size"].unique()


@callback(
    Output("graph-content", "figure"),
    [
        Input("column-radios", "value"),
        Input("further-value-dd", "value"),
        Input("further-checkbox", "value"),
        Input("column-checkboxes", "value"),
    ],
)
def update_graph(
    x_axis_further, further_filter_value, further_checkbox_value, series_names
):
    x_axis, further_filter_name = "sample_size", "splits"

    if x_axis_further == 2:
        x_axis, further_filter_name = "splits", "sample_size"

    further_series_values = (
        df[further_filter_name].unique()
        if "Show all" in further_checkbox_value
        else [further_filter_value]
    )

    fig = go.Figure()

    for further_series_value in further_series_values:
        dff = df[df[further_filter_name] == further_series_value].sort_values(by=x_axis)

        for series_name in series_names:
            fig.add_trace(
                go.Scatter(
                    x=dff[x_axis],
                    y=(
                        dff[series_name]
                        if series_name != "squadv2_f1"
                        else dff[series_name] / 100
                    ),
                    mode="lines+markers",
                    marker=dict(
                        size=10,
                        symbol="circle",
                    ),
                    name=(
                        f"{series_name}_{further_series_value}"
                        if len(further_series_values) > 1
                        else series_name
                    ),
                )
            )

    if "Show baselines" in further_checkbox_value:
        for series_name in series_names:
            fig.add_hline(
                y=(
                    baselines_df[series_name][0]
                    if series_name != "squadv2_f1"
                    else baselines_df[series_name][0] / 100
                ),
                line_width=3,
                line_dash="dash",
            )

    fig.update_layout(xaxis=dict(tickvals=dff[x_axis].unique()))
    return fig


if __name__ == "__main__":
    app.run(debug=True)
