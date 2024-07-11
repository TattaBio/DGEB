import math
import json
from pathlib import Path
import gradio as gr
from typing import List
import pandas as pd
import importlib.util
from pydantic import ValidationError, parse_obj_as

SIG_FIGS = 4

# HACK: very hacky way to import from parent directory, while avoiding needing all the deps of the parent package
tasks_path = "../dgeb/tasks/tasks.py"

# Load the module
spec = importlib.util.spec_from_file_location("tasks", tasks_path)
tasks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tasks)
TaskResult = tasks.TaskResult
GEBModel = tasks.GEBModel

# Assuming the class definitions provided above are complete and imported here


def format_num_params(param: int) -> str:
    # if the number of parameters is greater than 1 billion, display billion
    million = 1_000_000
    # billion = 1_000_000_000
    # if param >= billion:
    #     num_billions = int(param / 1_000_000_000)
    #     return f"{num_billions:}B"
    if param >= million:
        num_millions = int(param / 1_000_000)
        return f"{num_millions:}M"
    else:
        return f"{param:,}"


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


def load_results() -> List[TaskResult]:
    """
    Recursively load JSON files in ./submissions/** and return a list of TaskResult objects.
    """
    submissions_path = Path("./submissions")
    json_contents = load_json_files_from_directory(submissions_path)

    task_results_objects = []
    for content in json_contents:
        try:
            task_result = parse_obj_as(
                TaskResult, content
            )  # Using Pydantic's parse_obj_as for creating TaskResult objects
            task_results_objects.append(task_result)
        except ValidationError as e:
            print(f"Error parsing TaskResult object: {e}")
            raise e

    return task_results_objects


def task_results_to_dgeb_score(
    model: GEBModel, model_results: List[TaskResult]
) -> dict:
    best_scores_per_task = []
    modalities_seen = set()
    for task_result in model_results:
        modalities_seen.add(task_result.task.modality)
        assert (
            task_result.model.hf_name == model.hf_name
        ), f"Model names do not match, {task_result.model.hf_name} != {model.hf_name}"
        primary_metric_id = task_result.task.primary_metric_id
        scores = []
        # Get the primary score for each layer.
        for result in task_result.results:
            for metric in result.metrics:
                if metric.id == primary_metric_id:
                    scores.append(metric.value)
        best_score = max(scores)
        best_scores_per_task.append(best_score)

    assert (
        len(modalities_seen) == 1
    ), f"Multiple modalities found for model {model.hf_name}"
    # Calculate the average of the best scores for each task.
    assert len(best_scores_per_task) > 0, f"No tasks found for model {model.hf_name}"
    dgeb_score = sum(best_scores_per_task) / len(best_scores_per_task)
    return {
        "Task Name": "DGEB Score",
        "Task Category": "DGEB",
        "Model": model.hf_name,
        "Modality": list(modalities_seen)[0],
        "Num. Parameters (millions)": format_num_params(model.num_params),
        "Emb. Dimension": model.embed_dim,
        "Score": dgeb_score,
    }


def task_results_to_df(model_results: List[TaskResult]) -> pd.DataFrame:
    # Initialize an empty list to hold all rows of data
    data_rows = []
    all_models = {}
    for res in model_results:
        task = res.task
        model = res.model
        all_models[model.hf_name] = model
        print(f"Processing {task.display_name} for {model.hf_name}")
        for layer in res.results:
            total_layers = model.num_layers - 1
            mid_layer = math.ceil(total_layers / 2)
            if mid_layer == layer.layer_number:
                layer.layer_display_name = "mid"
            elif total_layers == layer.layer_number:
                layer.layer_display_name = "last"

            if layer.layer_display_name not in ["mid", "last"]:
                # calculate if the layer is mid or last
                print(
                    f"Layer {layer.layer_number} is not mid or last out of {total_layers}. Skipping"
                )
                continue
            else:
                # For each Metric in the Layer
                # pivoting the data so that each metric is a row
                metric_ids = []
                primary_metric_label = f"{task.primary_metric_id} (primary metric)"
                for metric in layer.metrics:
                    if task.primary_metric_id == metric.id:
                        metric_ids.append(primary_metric_label)
                    else:
                        metric_ids.append(metric.id)

                metric_values = [metric.value for metric in layer.metrics]
                zipped = zip(metric_ids, metric_values)
                # sort primary metric id first
                sorted_zip = sorted(
                    zipped,
                    key=lambda x: x[0] != primary_metric_label,
                )
                data_rows.append(
                    {
                        "Task Name": task.display_name,
                        "Task Category": task.type,
                        "Model": model.hf_name,
                        "Num. Parameters (millions)": format_num_params(
                            model.num_params
                        ),
                        "Emb. Dimension": model.embed_dim,
                        "Modality": task.modality,
                        "Layer": layer.layer_display_name,
                        **dict(sorted_zip),
                    }
                )
    for model_name, model in all_models.items():
        results_for_model = [
            res for res in model_results if res.model.hf_name == model_name
        ]
        assert len(results_for_model) > 0, f"No results found for model {model_name}"
        dgeb_score_record = task_results_to_dgeb_score(model, results_for_model)
        print(f'model {model.hf_name} dgeb score: {dgeb_score_record["Score"]}')
        data_rows.append(dgeb_score_record)
    print("Finished processing all results")
    df = pd.DataFrame(data_rows)
    return df


df = task_results_to_df(load_results())
image_path = "./DGEB_Figure.png"
with gr.Blocks() as demo:
    gr.Label("Diverse Genomic Embedding Benchmark", show_label=False, scale=2)
    gr.HTML(
        f"<img src='file/{image_path}' alt='DGEB Figure' style='border-radius: 0.8rem; width: 50%; margin-left: auto; margin-right: auto; margin-top:12px;'>"
    )
    gr.HTML(
        """
<div style='width: 50%; margin-left: auto; margin-right: auto; padding-bottom: 8px;text-align: center;'>
DGEB Leaderboard. To submit, refer to the <a href="https://github.com/TattaBio/DGEB/blob/leaderboard/README.md" target="_blank" style="text-decoration: underline">DGEB GitHub repository</a> Refer to the [DGEB paper](https://example.com) for details on metrics, tasks, and models.
</div>
"""
    )

    unique_categories = df["Task Category"].unique()
    # sort "DGEB" to the start
    unique_categories = sorted(unique_categories, key=lambda x: x != "DGEB")
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
                    # round all values to 4 decimal places
                    rounded_df = filtered_df.round(SIG_FIGS)

                    # calculate ranking column
                    # if in Overview tab, rank by average metric value
                    if task == "Overall":
                        # rank by average col
                        rounded_df["Rank"] = filtered_df["Average"].rank(
                            ascending=False
                        )
                    else:
                        avoid_cols = [
                            "Model",
                            "Emb. Dimension",
                            "Num. Parameters (millions)",
                            "Modality",
                            "Layer",
                        ]
                        rounded_df["Rank"] = (
                            rounded_df.drop(columns=avoid_cols, errors="ignore")
                            .sum(axis=1)
                            .rank(ascending=False)
                        )
                    # make Rank first column
                    cols = list(rounded_df.columns)
                    cols.insert(0, cols.pop(cols.index("Rank")))
                    rounded_df = rounded_df[cols]
                    # sort by rank
                    rounded_df = rounded_df.sort_values("Rank")
                    data_frame = gr.DataFrame(rounded_df)


demo.launch(allowed_paths=["."])
