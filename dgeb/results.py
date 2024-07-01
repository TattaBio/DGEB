from typing import List
from pydantic import BaseModel


class Metric(BaseModel):
    metric_id: str
    display_name: str
    display_value: float
    description: str


class TaskResult(BaseModel):
    task_id: str
    display_name: str
    description: str
    metrics: List[Metric]


class TaskResults(BaseModel):
    results_id: str
    description: str
    task_results: List[TaskResult]


# Example task results that conforms to above data specification
mock_task_results = {
    "results_id": "result_123",
    "description": "Overall results of the tasks",
    "task_results": [
        {
            "task_id": "task_1",
            "display_name": "Task 1",
            "description": "Description of Task 1",
            "metrics": [
                {
                    "metric_id": "metric_1",
                    "display_name": "Metric 1",
                    "display_value": "Value 1",
                    "description": "Description of Metric 1",
                },
                {
                    "metric_id": "metric_2",
                    "display_name": "Metric 2",
                    "display_value": "Value 2",
                    "description": "Description of Metric 2",
                },
            ],
        },
    ],
}
