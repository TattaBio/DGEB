"""
Given a directory of results, plot the benchmarks for each task as a bar chart and line chart.
"""

import argparse
import os
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from dgeb.geb import get_all_tasks, get_output_folder, get_tasks_by_name
from dgeb.tasks.tasks import TaskResult

ALL_TASKS = [task.metadata.id for task in get_all_tasks()]


def plot_benchmarks(
    results_dir,
    task_ids: Optional[list[str]] = None,
    output="benchmarks.png",
    model_substring=None,
):
    models = os.listdir(results_dir)
    all_results = []
    tasks = get_all_tasks() if task_ids is None else get_tasks_by_name(task_ids)
    for model_name in models:
        if model_substring is not None and all(
            substr not in model_name for substr in model_substring
        ):
            continue

        for task in tasks:
            if task.metadata.display_name == "NoOp Task":
                continue
            filepath = get_output_folder(model_name, task, results_dir, create=False)
            # if the file does not exist, skip
            if not os.path.exists(filepath):
                continue

            with open(filepath) as f:
                task_result = TaskResult.model_validate_json(f.read())
            num_params = task_result.model["num_params"]
            primary_metric_id = task_result.task.primary_metric_id
            main_scores = [
                metric.value
                for layer_result in task_result.results
                for metric in layer_result.metrics
                if metric.id == primary_metric_id
            ]
            best_score = max(main_scores)
            all_results.append(
                {
                    "task": task.metadata.display_name,
                    "model": model_name,
                    "num_params": num_params,
                    "score": best_score,
                }
            )

    results_df = pd.DataFrame(all_results)
    # order the models by ascending number of parameters
    results_df["num_params"] = results_df["num_params"].astype(int)
    results_df = results_df.sort_values(by="num_params")
    # number of tasks
    n_tasks = len(set(results_df["task"]))

    _, ax = plt.subplots(2, n_tasks, figsize=(5 * n_tasks, 10))

    for i, task in enumerate(set(results_df["task"])):
        if n_tasks > 1:
            sns.barplot(
                x="model",
                y="score",
                data=results_df[results_df["task"] == task],
                ax=ax[0][i],
            )
            ax[0][i].set_title(task)
            # rotate the x axis labels

            for tick in ax[0][i].get_xticklabels():
                tick.set_rotation(90)
        else:
            sns.barplot(
                x="model",
                y="score",
                data=results_df[results_df["task"] == task],
                ax=ax[0],
            )
            ax[0].set_title(task)
            # rotate the x axis labels
            for tick in ax[0].get_xticklabels():
                tick.set_rotation(90)

    # make a line graph with number of parameters on x axis for each task in the second row of figures
    for i, task in enumerate(set(results_df["task"])):
        if n_tasks > 1:
            sns.lineplot(
                x="num_params",
                y="score",
                data=results_df[results_df["task"] == task],
                ax=ax[1][i],
            )
            ax[1][i].set_title(task)
            ax[1][i].set_xlabel("Number of parameters")
        else:
            sns.lineplot(
                x="num_params",
                y="score",
                data=results_df[results_df["task"] == task],
                ax=ax[1],
            )
            ax[1].set_title(task)
            ax[1].set_xlabel("Number of parameters")

    plt.tight_layout()
    plt.savefig(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--results_dir",
        type=str,
        default="results",
        help="Directory containing the results of the benchmarking",
    )
    parser.add_argument(
        "-t",
        "--tasks",
        type=lambda s: [item for item in s.split(",")],
        default=None,
        help=f"Comma separated list of tasks to plot. Choose from {ALL_TASKS} or do not specify to plot all tasks. ",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="benchmarks.png",
        help="Output file for the plot",
    )
    parser.add_argument(
        "--model_substring",
        type=lambda s: [item for item in s.split(",")],
        default=None,
        help="Comma separated list of model substrings. Only plot results for models containing this substring",
    )
    args = parser.parse_args()

    plot_benchmarks(args.results_dir, args.tasks, args.output, args.model_substring)
