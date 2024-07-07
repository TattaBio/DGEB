import gradio as gr
from typing import List
from lib.models import TaskResults, gen_data
import pandas as pd


# Example task results that conforms to above data specification
mock_results = gen_data(n_models=8, n_tasks=50)


def task_results_to_df(model_results: List[TaskResults]) -> pd.DataFrame:
    # Initialize an empty list to hold all rows of data
    data_rows = []
    for res in model_results:
        task = res.task
        model = res.model
        for layer in res.results:
            if layer.layer_display_name not in ["mid", "last"]:
                continue
            else:
                # For each Metric in the Layer
                # pivoting the data so that each metric is a row
                metric_ids = [metric.metric_id for metric in layer.metrics]
                metric_values = [metric.value for metric in layer.metrics]
                data_rows.append(
                    {
                        "Task Name": task.display_name,
                        "Task Category": task.category,
                        "Model": model.display_name,
                        "Layer": layer.layer_display_name,
                        **dict(zip(metric_ids, metric_values)),
                    }
                )
    categories = set([res.task.category for res in model_results])
    all_layers = set(
        [
            (res.model.display_name, layer.layer_display_name)
            for res in model_results
            for layer in res.results
        ]
    )
    for model_name, layer_name in all_layers:
        for category in categories:
            data_rows.append(
                {
                    "Task Name": "Overall",
                    "Task Category": category,
                    "Model": model_name,
                    "Layer": layer_name,
                    "average": -1,
                }
            )
    df = pd.DataFrame(data_rows)
    return df


df = task_results_to_df(mock_results)
with gr.Blocks() as demo:

    def update_df(model_search: str) -> pd.DataFrame:
        # reset to original df, don't want to filter on filtered df
        # because once filtered, it will keep getting smaller
        filtered_df: pd.DataFrame = df.copy()
        if model_search:
            filtered_df = df[df["Model"].str.contains(model_search, case=False)]
        return filtered_df

    gr.Markdown(
        """
        DGEB Leaderboard. To submit, refer to the <a href="https://github.com/TattaBio/GEB/blob/leaderboard/README.md" target="_blank" style="text-decoration: underline">DGEB GitHub repository</a> 🤗 Refer to the [GEB paper](https://example.com) for details on metrics, tasks, and models.
        """
    )

    with gr.Row():
        model_search = gr.Textbox(
            label="Models", placeholder=" 🔍 Search for a model and press enter..."
        )
    unique_categories = df["Task Category"].unique()
    for category in unique_categories:
        with gr.Tab(label=category):
            unique_tasks_in_category = df[df["Task Category"] == category][
                "Task Name"
            ].unique()
            # sort "Overall" to the start
            unique_tasks_in_category = sorted(
                unique_tasks_in_category, key=lambda x: x != "Overall"
            )
            for task in unique_tasks_in_category:
                with gr.Tab(label=task):
                    columns_to_hide = ["Task Name", "Task Category"]
                    # get rows where Task Name == task and Task Category == category
                    filtered_df = (
                        df[
                            (df["Task Name"] == task)
                            & (df["Task Category"] == category)
                        ].drop(columns=columns_to_hide)
                    ).dropna(
                        axis=1, how="all"
                    )  # drop all NaN columns for Overall tab

                    data_frame = gr.DataFrame(filtered_df)
                    model_search.change(
                        update_df, inputs=[model_search], outputs=data_frame
                    )


demo.launch()
