"""
Biological sequences are clustered and performance is determined by how well clustering matches assigned labels.
"""

import logging
from collections import defaultdict

from dgeb.evaluators import ClusteringEvaluator
from dgeb.modality import Modality
from dgeb.models import BioSeqTransformer
from dgeb.tasks import Dataset, Task, TaskMetadata, TaskResult

logger = logging.getLogger(__name__)


def run_clustering_task(model: BioSeqTransformer, metadata: TaskMetadata) -> TaskResult:
    """Evaluate clustering task. Utilizes the ClusteringEvaluator."""
    if len(metadata.datasets) != 1:
        raise ValueError("Clustering tasks require 1 dataset.")
    ds = metadata.datasets[0].load()["train"]
    embeds = model.encode(ds["Sequence"])
    layer_results = defaultdict(dict)
    for i, layer in enumerate(model.layers):
        labels = ds["Label"]
        evaluator = ClusteringEvaluator(embeds[:, i], labels)
        layer_results["layers"][layer] = evaluator()
        logger.info(
            f"Layer: {layer}, {metadata.display_name} results: {layer_results['layers'][layer]}"
        )
    return TaskResult.from_dict(metadata, layer_results, model.metadata)


class RNAclustering(Task):
    metadata = TaskMetadata(
        id="ecoli_rna_clustering",
        display_name="E.coli RNA Clustering",
        description="Evaluate on RNA clustering task for sRNA/tRNA/rRNA segments in E.coli K-12.",
        type="clustering",
        modality=Modality.DNA,
        datasets=[
            Dataset(
                path="tattabio/e_coli_rnas",
                revision="4c134bb4bdb2b0ef1d59fe10797efdfeaf318de6",
            )
        ],
        primary_metric_id="v_measure",
    )

    def run(self, model: BioSeqTransformer) -> TaskResult:
        return run_clustering_task(model, self.metadata)


class MopBClustering(Task):
    metadata = TaskMetadata(
        id="mopb_clustering",
        display_name="MopB Clustering",
        description="Evaluate on MopB clustering task.",
        type="clustering",
        modality=Modality.PROTEIN,
        datasets=[
            Dataset(
                path="tattabio/mopb_clustering",
                revision="eed4bfff9c5bd2dc2500c50757bfcb90425d999a",
            )
        ],
        primary_metric_id="v_measure",
    )

    def run(self, model: BioSeqTransformer) -> TaskResult:
        return run_clustering_task(model, self.metadata)
