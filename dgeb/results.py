# from importlib.metadata import version

from typing import List, Literal, Union, Optional
from pydantic import BaseModel


class Metric(BaseModel):
    id: str
    display_name: str
    description: Optional[str]
    value: float


class LayerResult(BaseModel):
    layer_number: int
    layer_display_name: str
    metrics: List[Metric]


class Dataset(BaseModel):
    path: str
    revision: str


class Task(BaseModel):
    id: str
    display_name: str
    description: str
    modality: Union[Literal["dna"], Literal["protein"]]
    type: Union[
        Literal["bigene_mining"],
        Literal["eds"],
        Literal["pair_classification"],
        Literal["classification"],
        Literal["clustering"],
        Literal["retrieval"],
    ]
    datasets: List[Dataset]
    primary_metric_id: str


# results.py
class TaskResults(BaseModel):
    dgeb_version: str
    task: Task
    model: GEBModel
    results: List[LayerResult]
