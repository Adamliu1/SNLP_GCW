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
    html.P("x axis", style={"font-weight": "bold"}),
    dcc.RadioItems(
        id="x-axis-radios",
        options=[
            {"label": "sample size", "value": 0},
            {"label": "splits", "value": 1},
            {"label": "epoch count", "value": 2},
        ],
        value=0,
        labelStyle={"display": "block"},
    ),
    html.P("sample size", style={"font-weight": "bold"}),
    dcc.Checklist(
        id="sample-size-checkbox",
        options=["Show all"],
        value=[],
        labelStyle={"width": "50%", "display": "block"},
    ),
    dcc.Dropdown(
        id="sample-size-dd",
        options=df.sample_size.unique(),
        value=df.sample_size.unique()[0],
    ),
    html.P("splits", style={"font-weight": "bold"}),
    dcc.Checklist(
        id="splits-checkbox",
        options=["Show all"],
        value=[],
        labelStyle={"width": "50%", "display": "block"},
    ),
    dcc.Dropdown(
        id="splits-dd",
        options=df.splits.unique(),
        value=df.splits.unique()[0],
    ),
    html.P("epoch count", style={"font-weight": "bold"}),
    dcc.Checklist(
        id="epoch-count-checkbox",
        options=["Show all"],
        value=[],
        labelStyle={"width": "50%", "display": "block"},
    ),
    dcc.Dropdown(
        id="epoch-count-dd",
        options=df.epoch_count.unique(),
        value=df.epoch_count.unique()[0],
    ),
    html.P("series to display:", style={"font-weight": "bold"}),
    dcc.Checklist(
        id="baselines-checkbox",
        options=["Show baselines"],
        value=[],
        labelStyle={"width": "50%", "display": "block"},
    ),
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
    Output("sample-size-dd", "disabled"),
    Output("splits-dd", "disabled"),
    Output("epoch-count-dd", "disabled"),
    Output("sample-size-checkbox", "options"),
    Output("splits-checkbox", "options"),
    Output("epoch-count-checkbox", "options"),
    Input("x-axis-radios", "value"),
    Input("sample-size-checkbox", component_property="value"),
    Input("splits-checkbox", component_property="value"),
    Input("epoch-count-checkbox", component_property="value"),
)
def select_x_axis(x_axis, sample_size_cbox, splits_cbox, epoch_count_cbox):
    output_dd = [False, False, False]
    output_cbox = [False, False, False]
    output_cbox[x_axis] = True
    output_dd[x_axis] = True
    if "Show all" in sample_size_cbox:
        output_dd[0] = True
    if "Show all" in splits_cbox:
        output_dd[1] = True
    if "Show all" in epoch_count_cbox:
        output_dd[2] = True
    return (
        *output_dd,
        [{"label": "Show all", "value": "Show all", "disabled": output_cbox[0]}],
        [{"label": "Show all", "value": "Show all", "disabled": output_cbox[1]}],
        [{"label": "Show all", "value": "Show all", "disabled": output_cbox[2]}],
    )


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


@callback(
    Output("graph-content", "figure"),
    [
        Input("x-axis-radios", "value"),
        Input("sample-size-dd", "value"),
        Input("splits-dd", "value"),
        Input("epoch-count-dd", "value"),
        Input("sample-size-checkbox", "value"),
        Input("splits-checkbox", "value"),
        Input("epoch-count-checkbox", "value"),
        Input("baselines-checkbox", "value"),
        Input("column-checkboxes", "value"),
    ],
)
def update_graph(
    x_axis, ss_dd, s_dd, ec_dd, ss_cbox, s_cbox, ec_cbox, baselines_cbox, series_names
):
    filters_dict = {}

    if x_axis == 0:
        x_axis_name = "sample_size"
    elif x_axis == 1:
        x_axis_name = "splits"
    else:
        x_axis_name = "epoch_count"

    filters_dict["sample_size"] = (
        df.sample_size.unique() if "Show all" in ss_cbox else [ss_dd]
    )
    filters_dict["splits"] = df.splits.unique() if "Show all" in s_cbox else [s_dd]
    filters_dict["epoch_count"] = (
        df.epoch_count.unique() if "Show all" in ec_cbox else [ec_dd]
    )

    del filters_dict[x_axis_name]

    fig = go.Figure()
    filter1_name, filter1_values = list(filters_dict.items())[0]
    filter2_name, filter2_values = list(filters_dict.items())[1]

    for val1 in filter1_values:
        for val2 in filter2_values:
            dff = df[
                (df[filter1_name] == val1) & (df[filter2_name] == val2)
            ].sort_values(by=x_axis_name)

            for series_name in series_names:
                fig.add_trace(
                    go.Scatter(
                        x=dff[x_axis_name],
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
                        name=(f"{series_name}_{val1}_{val2}"),
                    )
                )

    if "Show baselines" in baselines_cbox:
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

    fig.update_layout(xaxis=dict(tickvals=dff[x_axis_name].unique()))
    return fig


if __name__ == "__main__":
    app.run(debug=True)
