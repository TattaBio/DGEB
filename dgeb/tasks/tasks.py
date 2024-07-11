"""Task abstract class for evaluation and results."""

import logging
from typing import List, Literal, Optional, Any
from importlib.metadata import version
from enum import Enum
import datasets
from pydantic import BaseModel, model_validator
from abc import ABC, abstractmethod


# HACK: if Modality is not defined, then import it from modality.py
try:
    from ..modality import Modality
except:
    # if not, super hack to get the leaderboard working.
    # SHOULD MATCH the code exactly in modality.py
    # can we read the file and run that code?
    from enum import Enum

    class Modality(Enum):
        """Data modality, either DNA or protein sequence."""

        PROTEIN = "protein"
        DNA = "dna"


logging.basicConfig(level=logging.INFO)

TaskType = Literal[
    "classification",
    "pair_classification",
    "clustering",
    "eds",
    "bigene_mining",
    "retrieval",
]


class TaskMetric(BaseModel):
    id: str
    display_name: str
    description: Optional[str] = None
    value: float = 0.0


class LayerResult(BaseModel):
    layer_number: int
    layer_display_name: str
    metrics: List[TaskMetric]


class GEBModel(BaseModel):
    hf_name: str
    num_layers: int
    num_params: int
    embed_dim: int


class Dataset(BaseModel):
    path: str
    revision: str

    def load(self) -> datasets.DatasetDict:
        ds = datasets.load_dataset(self.path, revision=self.revision)
        if not isinstance(ds, datasets.DatasetDict):
            raise ValueError(
                f"Dataset {self.path} is not a datasets.DatasetDict object."
            )
        return ds


class TaskMetadata(BaseModel):
    id: str
    display_name: str
    description: str
    modality: Modality
    type: TaskType
    # List of datasets used by the task.
    # Each dataset is a dict of all arguments to pass to `datasets.load_dataset()`.
    datasets: List[Dataset]
    primary_metric_id: str


# tasks.py
class TaskResult(BaseModel):
    dgeb_version: str
    task: "TaskMetadata"
    # TODO: Convert model to ModelMetadata
    model: GEBModel
    results: List[LayerResult]

    @model_validator(mode="after")
    def check_valid_primary_metric(self):
        for result in self.results:
            if all(
                metric.id != self.task.primary_metric_id for metric in result.metrics
            ):
                raise ValueError(
                    f"Primary metric {self.task.primary_metric_id} not found in results.metrics"
                )
        return self

    @staticmethod
    def from_dict(
        task_metadata: "TaskMetadata",
        layer_results: LayerResult,
        model_metadata: GEBModel,
    ):
        return TaskResult(
            dgeb_version=version("dgeb"),
            task=task_metadata,
            model=model_metadata,
            results=list(
                LayerResult(
                    layer_number=int(layer),
                    layer_display_name=str(layer),
                    metrics=[
                        TaskMetric(id=metric, display_name=metric, value=value)
                        for metric, value in metrics.items()
                    ],
                )
                for layer, metrics in layer_results["layers"].items()
            ),
        )


# move to model.py?
class Task(ABC):
    metadata: TaskMetadata

    # using Any instead of "BioSeqTransformer" to avoid installing all deps in leaderboard
    @abstractmethod
    def run(self, model: Any, layers: Optional[List[int]] = None) -> TaskResult:
        pass
