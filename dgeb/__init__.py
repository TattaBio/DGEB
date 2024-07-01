from dgeb.dgeb import (
    DGEB,
    get_all_model_names,
    get_all_task_names,
    get_all_tasks,
    get_model,
    get_output_folder,
    get_tasks_by_modality,
    get_tasks_by_name,
)
from dgeb.modality import Modality
from dgeb.tasks.tasks import TaskResult

# importing without setting `__all__` produces a Ruff error:
#   "imported but unused; consider removing, adding to __all__, or using a redundant alias RuffF401"
# See https://docs.astral.sh/ruff/rules/unused-import/#why-is-this-bad
__all__ = [
    "DGEB",
    "get_all_tasks",
    "get_all_task_names",
    "get_tasks_by_name",
    "get_tasks_by_modality",
    "get_all_model_names",
    "get_model",
    "get_output_folder",
    "TaskResult",
    "Modality",
]
