"""
Pair classification tasks evaluating distances between functionally relevant gene pairs.
For instance, distance thresholds distinguish between co-transcribed and non-co-transcribed gene pairs.
"""

import logging
from collections import defaultdict

from geb.evaluators import PairClassificationEvaluator
from geb.modality import Modality
from geb.models import BioSeqTransformer
from geb.tasks.tasks import Dataset, Task, TaskMetadata, TaskResult

from ..eval_utils import paired_dataset

logger = logging.getLogger(__name__)


def run_pair_classification_task(
    model: BioSeqTransformer, metadata: TaskMetadata
) -> TaskResult:
    """Evaluate pair classification task. Utilizes the PairClassificationEvaluator."""
    if len(metadata.datasets) != 1:
        raise ValueError("Pair classification tasks require 1 dataset.")
    ds = metadata.datasets[0].load()["train"]
    embeds = model.encode(ds["Sequence"])
    layer_results = defaultdict(dict)
    for i, layer in enumerate(model.layers):
        labels = ds["Label"]
        embeds1, embeds2, labels = paired_dataset(labels, embeds[:, i])
        evaluator = PairClassificationEvaluator(embeds1, embeds2, labels)
        layer_results["layers"][layer] = evaluator()
        logger.info(
            f"Layer: {layer}, {metadata.display_name} classification results: {layer_results['layers'][layer]}"
        )
    return TaskResult.from_dict(metadata, layer_results, model.metadata)


class EcoliOperon(Task):
    metadata = TaskMetadata(
        id="ecoli_operonic_pair",
        display_name="E.coli Operonic Pair",
        description="Evaluate on E.coli K-12 operonic pair classification task.",
        type="pair_classification",
        modality=Modality.PROTEIN,
        datasets=[
            Dataset(
                path="tattabio/ecoli_operonic_pair",
                revision="main",
            )
        ],
        primary_metric_id="top_ap",
    )

    def run(self, model: BioSeqTransformer) -> TaskResult:
        return run_pair_classification_task(model, self.metadata)


class CyanoOperonPair(Task):
    metadata = TaskMetadata(
        id="cyano_operonic_pair",
        display_name="Cyano Operonic Pair",
        description="Evaluate on Cyano operonic pair classification task.",
        type="pair_classification",
        modality=Modality.PROTEIN,
        datasets=[
            Dataset(
                path="tattabio/cyano_operonic_pair",
                revision="main",
            )
        ],
        primary_metric_id="top_ap",
    )

    def run(self, model: BioSeqTransformer) -> TaskResult:
        return run_pair_classification_task(model, self.metadata)


class VibrioOperonPair(Task):
    metadata = TaskMetadata(
        id="vibrio_operonic_pair",
        display_name="Vibrio Operonic Pair",
        description="Evaluate on Vibrio operonic pair classification task.",
        type="pair_classification",
        modality=Modality.PROTEIN,
        datasets=[
            Dataset(
                path="tattabio/vibrio_operonic_pair",
                revision="main",
            )
        ],
        primary_metric_id="top_ap",
    )

    def run(self, model: BioSeqTransformer) -> TaskResult:
        return run_pair_classification_task(model, self.metadata)
