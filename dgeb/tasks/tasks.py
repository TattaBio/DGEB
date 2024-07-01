"""Task functions for evaluation.
# TODO: Add dataset revisions.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Literal, Optional

import datasets
from pydantic import BaseModel, model_validator

from ..modality import Modality
from ..models import BioSeqTransformer

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


class TaskResult(BaseModel):
    task: "TaskMetadata"
    # TODO: Convert model to ModelMetadata
    model: Dict[str, Any]
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
        layer_results: Dict[str, Any],
        model_metadata: Dict[str, Any],
    ):
        return TaskResult(
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


class Task(ABC):
    metadata: TaskMetadata

    @abstractmethod
    def run(
        self, model: BioSeqTransformer, layers: Optional[List[int]] = None
    ) -> TaskResult:
        pass


class noop(Task):
    metadata = TaskMetadata(
        id="noop",
        display_name="NoOp Task",
        description="This task is used for testing and does nothing.",
        type="classification",
        modality=Modality.PROTEIN,
        datasets=[
            Dataset(
                path="",
                revision="main",
            )
        ],
        primary_metric_id="f1",
    )

    def run(self, model: BioSeqTransformer) -> TaskResult:
        return TaskResult.from_dict(
            self.metadata,
            {"layers": {32: {"accuracy": 0.5, "f1": 0.5}}},
            model.metadata,
        )
