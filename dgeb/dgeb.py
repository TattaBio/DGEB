import logging
import os
import traceback
from itertools import chain
from typing import Any, List

from rich.console import Console

from .eval_utils import set_all_seeds
from .modality import Modality
from .models import BioSeqTransformer
from .tasks.tasks import Task

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DGEB:
    """GEB class to run the evaluation pipeline."""

    def __init__(self, tasks: List[type[Task]], seed: int = 42):
        self.tasks = tasks
        set_all_seeds(seed)

    def print_selected_tasks(self):
        """Print the selected tasks."""
        console = Console()
        console.rule("[bold]Selected Tasks\n", style="grey15")
        for task in self.tasks:
            prefix = "    - "
            name = f"{task.metadata.display_name}"
            category = f", [italic grey39]{task.metadata.type}[/]"
            console.print(f"{prefix}{name}{category}")
        console.print("\n")

    def run(
        self,
        model,  # type encoder
        output_folder: str = "results",
    ):
        """Run the evaluation pipeline on the selected tasks.

        Args:
            model: Model to be used for evaluation
            output_folder: Folder where the results will be saved. Default to 'results'. Where it will save the results in the format:
                `{output_folder}/{model_name}/{model_revision}/{task_name}.json`.

        Returns:
            A list of MTEBResults objects, one for each task evaluated.
        """
        # Run selected tasks
        self.print_selected_tasks()
        results = []

        for task in self.tasks:
            logger.info(
                f"\n\n********************** Evaluating {task.metadata.display_name} **********************"
            )

            try:
                result = task().run(model)
            except Exception as e:
                logger.error(e)
                logger.error(traceback.format_exc())
                logger.error(f"Error running task {task}")
                continue

            results.append(result)

            save_path = get_output_folder(model.hf_name, task, output_folder)
            with open(save_path, "w") as f_out:
                f_out.write(result.model_dump_json(indent=2))
        return results


def get_model(model_name: str, **kwargs: Any) -> type[BioSeqTransformer]:
    all_names = get_all_model_names()
    for cls in BioSeqTransformer.__subclasses__():
        if model_name in cls.MODEL_NAMES:
            return cls(model_name, **kwargs)
    raise ValueError(f"Model {model_name} not found in {all_names}.")


def get_all_model_names() -> List[str]:
    return list(
        chain.from_iterable(
            cls.MODEL_NAMES for cls in BioSeqTransformer.__subclasses__()
        )
    )


def get_all_task_names() -> List[str]:
    return [task.metadata.id for task in get_all_tasks()]


def get_tasks_by_name(tasks: List[str]) -> List[type[Task]]:
    return [_get_task(task) for task in tasks]


def get_tasks_by_modality(modality: Modality) -> List[type[Task]]:
    return [task for task in get_all_tasks() if task.metadata.modality == modality]


def get_all_tasks() -> List[type[Task]]:
    return Task.__subclasses__()


def _get_task(task_name: str) -> type[Task]:
    logger.info(f"Getting task {task_name}")
    for task in get_all_tasks():
        if task.metadata.id == task_name:
            return task

    raise ValueError(
        f"Task {task_name} not found, available tasks are: {[task.metadata.id for task in get_all_tasks()]}"
    )


def get_output_folder(
    model_hf_name: str, task: type[Task], output_folder: str, create: bool = True
):
    output_folder = os.path.join(output_folder, os.path.basename(model_hf_name))
    # create output folder if it does not exist
    if create and not os.path.exists(output_folder):
        os.makedirs(output_folder)
    return os.path.join(
        output_folder,
        f"{task.metadata.id}.json",
    )
