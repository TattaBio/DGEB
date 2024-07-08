import json
from pathlib import Path
import gradio as gr
from typing import List
import pandas as pd
import importlib.util
from pydantic import ValidationError, parse_obj_as

# HACK: very hacky way to import from parent directory, while avoiding needing all the deps of the parent package
results_path = "../dgeb/results.py"

# Load the module
spec = importlib.util.spec_from_file_location("results", results_path)
results = importlib.util.module_from_spec(spec)
spec.loader.exec_module(results)
TaskResults = results.TaskResults

# Assuming the class definitions provided above are complete and imported here


def load_json_files_from_directory(directory_path: Path) -> List[dict]:
    """
    Recursively load all JSON files within the specified directory path.

    :param directory_path: Path to the directory to search for JSON files.
    :return: List of dictionaries loaded from JSON files.
    """
    json_files_content = []
    for json_file in directory_path.rglob("*.json"):  # Recursively find all JSON files
        try:
            with open(json_file, "r", encoding="utf-8") as file:
                json_content = json.load(file)
                json_files_content.append(json_content)
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
    return json_files_content


def load_results() -> List[TaskResults]:
    """
    Recursively load JSON files in ./submissions/** and return a list of TaskResults objects.
    """
    submissions_path = Path("./submissions")
    json_contents = load_json_files_from_directory(submissions_path)

    task_results_objects = []
    for content in json_contents:
        try:
            task_result = parse_obj_as(
                TaskResults, content
            )  # Using Pydantic's parse_obj_as for creating TaskResults objects
            task_results_objects.append(task_result)
        except ValidationError as e:
            print(f"Error parsing TaskResults object: {e}")
            raise e

    return task_results_objects


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
                        "Model": model.hf_name,
                        "Layer": layer.layer_display_name,
                        **dict(zip(metric_ids, metric_values)),
                    }
                )
    categories = set([res.task.type for res in model_results])
    all_layers = set(
        [
            (res.model.hf_name, layer.layer_display_name)
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


df = task_results_to_df(load_results())
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
                    ).dropna(axis=1, how="all")  # drop all NaN columns for Overall tab

                    data_frame = gr.DataFrame(filtered_df)
                    model_search.change(
                        update_df, inputs=[model_search], outputs=data_frame
                    )


demo.launch()
