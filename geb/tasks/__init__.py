# ruff: noqa: F403

from .tasks import Task
from .eds_tasks import *
from .pair_classification_tasks import *
from .retrieval_tasks import *
from .classification_tasks import *
from .clustering_tasks import *
from .bigene_mining_tasks import *

__all__ = [
    "Task",
]
