import random
from typing import List, Literal, Union
from .utils import rand_string

from pydantic import BaseModel


class Metric(BaseModel):
    metric_id: str
    display_name: str
    value: float
    description: str


class LayerResult(BaseModel):
    layer_number: int
    layer_display_name: str
    metrics: List[Metric]


class Task(BaseModel):
    task_id: int
    display_name: str
    description: str
    modalities: Union[Literal["DNA"], Literal["Protein"]]
    category: Union[
        Literal["Bitext Mining"],
        Literal["EDS"],
        Literal["Pair Classification"],
        Literal["Classification"],
        Literal["Clustering"],
        Literal["Retrieval"],
    ]
    dataset_revision: str
    primary_metric_id: str


class GEBModel(BaseModel):
    display_name: str
    num_layers: int
    num_params: int
    layers: List[Union[Literal["first"], Literal["mid"], Literal["last"], int]]


class TaskResults(BaseModel):
    task: Task
    model: GEBModel
    results: List[LayerResult]


def gen_data(n_models: int, n_tasks: int) -> List[TaskResults]:
    tasks = []
    models = []
    task_results = []

    for i in range(n_tasks):
        task = Task(
            task_id=i,
            display_name=f"Task {i}",
            description=rand_string(100),
            modalities=random.choice(["DNA", "Protein"]),
            category=random.choice(
                [
                    "Bitext Mining",
                    "EDS",
                    "Pair Classification",
                    "Classification",
                    "Clustering",
                    "Retrieval",
                ]
            ),
            dataset_revision=f"revision_{i}",
            primary_metric_id=f"metric_{i}",
        )
        tasks.append(task)

    for i in range(n_models):
        model = GEBModel(
            display_name=f"Model {i}",
            num_layers=random.randint(1, 10),
            num_params=random.randint(1000, 10000),
            layers=random.choices(["mid", "last"], k=2),
        )
        models.append(model)

    for task in tasks:
        for model in models:
            layer_results = []
            for layer in model.layers:
                metrics = [
                    Metric(
                        metric_id=f"metric_{j}",
                        display_name=f"Metric {j}",
                        value=random.uniform(0, 100),
                        description=f"Description of metric {j}",
                    )
                    for j in range(random.randint(1, 5))
                ]
                layer_result = LayerResult(
                    layer_number=(
                        layer
                        if isinstance(layer, int)
                        else random.randint(0, model.num_layers - 1)
                    ),
                    layer_display_name=f"{layer}",
                    metrics=metrics,
                )
                layer_results.append(layer_result)
            task_result = TaskResults(task=task, model=model, results=layer_results)
            task_results.append(task_result)

    return task_results
