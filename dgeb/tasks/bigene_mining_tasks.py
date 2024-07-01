"""
Bigene mining tasks are analogous to bitext matching tasks, but for genes.
Cosine similarity is used to mine genes of related functions from different organisms.
"""

import logging
from collections import defaultdict

from geb.evaluators import BiGeneMiningEvaluator
from geb.modality import Modality
from geb.models import BioSeqTransformer
from geb.tasks.tasks import Dataset, Task, TaskMetadata, TaskResult

logger = logging.getLogger(__name__)


def run_bigene_mining_tasks(
    model: BioSeqTransformer, metadata: TaskMetadata, top_k: int = 1
) -> TaskResult:
    """Evaluate bigene mining task. Utilizes the BiGeneMiningEvaluator."""
    if len(metadata.datasets) != 1:
        raise ValueError("BiGeneMining tasks require 1 dataset.")
    ds = metadata.datasets[0].load()["train"]
    layer_results = defaultdict(dict)
    embeds1 = model.encode(ds["Seq1"])
    embeds2 = model.encode(ds["Seq2"])
    for i, layer in enumerate(model.layers):
        evaluator = BiGeneMiningEvaluator(embeds1[:, i], embeds2[:, i], top_k=top_k)
        layer_results["layers"][layer] = evaluator()
        logger.info(
            f"Layer: {layer}, {metadata.display_name} matching results: {layer_results['layers'][layer]}"
        )
    return TaskResult.from_dict(metadata, layer_results, model.metadata)


class BacArchBiGeneMining(Task):
    metadata = TaskMetadata(
        id="bacarch_bigene",
        display_name="BacArch BiGene",
        description="Evaluate on BacArch bigene matching task between bacterial (E.coli K-12) proteins and archaeal (Sulfolobus acidocaldarius DSM 639) proteins.",
        type="bigene_mining",
        modality=Modality.PROTEIN,
        datasets=[
            Dataset(
                path="tattabio/bac_arch_bigene",
                revision="main",
            )
        ],
        primary_metric_id="f1",
    )

    def run(self, model: BioSeqTransformer) -> TaskResult:
        return run_bigene_mining_tasks(model, self.metadata)


class ModACParalogyBiGeneMining(Task):
    # ModAC Paralogy matching with top_k=1 is too strict (most models have accuracy < 0.1%)
    # Instead use recall@5 as the main metric.
    TOP_K = 5

    metadata = TaskMetadata(
        id="modac_paralogy_bigene",
        display_name="ModAC Paralogy BiGene",
        description="Evaluate on paralogy bitext matching task between paralogous protein (ModA and ModC).",
        type="bigene_mining",
        modality=Modality.PROTEIN,
        datasets=[
            Dataset(
                path="tattabio/modac_paralogy_bigene",
                revision="main",
            )
        ],
        primary_metric_id="recall_at_5",
    )

    def run(self, model: BioSeqTransformer) -> TaskResult:
        return run_bigene_mining_tasks(model, self.metadata, top_k=self.TOP_K)
