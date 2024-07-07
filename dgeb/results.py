from typing import List, Literal, Union
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


mock_task_results = TaskResults(
    **{
        "task": {
            "task_id": 1,
            "display_name": "Task 1",
            "description": "Description 1",
            "modalities": "DNA",
            "category": "Bitext Mining",
            "dataset_revision": "1.0",
            "primary_metric_id": "accuracy",
        },
        "model": {
            "display_name": "Model 1",
            "num_layers": 3,
            "num_params": 1000,
            "layers": ["first", "mid", "last"],
        },
        "results": [
            {
                "layer_number": 0,
                "layer_display_name": "first",
                "metrics": [
                    {
                        "metric_id": "accuracy",
                        "display_name": "Accuracy",
                        "value": 0.9,
                        "description": "Accuracy of the model",
                    }
                ],
            },
            {
                "layer_number": 1,
                "layer_display_name": "mid",
                "metrics": [
                    {
                        "metric_id": "accuracy",
                        "display_name": "Accuracy",
                        "value": 0.8,
                        "description": "Accuracy of the model",
                    }
                ],
            },
            {
                "layer_number": 2,
                "layer_display_name": "last",
                "metrics": [
                    {
                        "metric_id": "accuracy",
                        "display_name": "Accuracy",
                        "value": 0.7,
                        "description": "Accuracy of the model",
                    }
                ],
            },
        ],
    }
)
