"""
Retrieval tasks find functionally relevant genes in a corpus of genes based on a query gene.
Typically corpus is derived from a different phylogenetic group than the query genes.
"""

import logging
from collections import defaultdict

from geb.evaluators import RetrievalEvaluator
from geb.modality import Modality
from geb.models import BioSeqTransformer
from geb.tasks.tasks import Dataset, Task, TaskMetadata, TaskResult

logger = logging.getLogger(__name__)


def run_retrieval_task(model: BioSeqTransformer, metadata: TaskMetadata) -> TaskResult:
    """Evaluate retrieval task. Utilizes the Retrieval evaluator."""
    if len(metadata.datasets) != 2:
        raise ValueError("Retrieval tasks require 3 datasets: corpus, query and qrels.")
    corpus_ds = metadata.datasets[0].load()["train"]
    query_ds = metadata.datasets[0].load()["test"]
    qrels = metadata.datasets[1].load()
    corpus_embeds = model.encode(corpus_ds["Sequence"])
    query_embeds = model.encode(query_ds["Sequence"])
    qrels_dict = defaultdict(dict)

    def qrels_dict_init(row):
        qrels_dict[str(row["query_id"])][str(row["corpus_id"])] = int(row["fuzz_ratio"])

    # Populate `qrels_dict` from the dataset.
    # See https://github.com/cvangysel/pytrec_eval for qrels format.
    qrels.map(qrels_dict_init)
    qrels = qrels_dict
    layer_results = defaultdict(dict)
    for i, layer in enumerate(model.layers):
        evaluator = RetrievalEvaluator(
            corpus_embeds[:, i],
            query_embeds[:, i],
            corpus_ds["Entry"],
            query_ds["Entry"],
            qrels,
        )
        layer_results["layers"][layer] = evaluator()
        logger.info(
            f"Layer: {layer}, Retrieval results: {layer_results['layers'][layer]}"
        )
    return TaskResult.from_dict(metadata, layer_results, model.metadata)


class ArchRetrieval(Task):
    metadata = TaskMetadata(
        id="arch_retrieval",
        display_name="Arch Retrieval",
        description="Retrieves bacterial proteins with similar swissprot annotations to a query archaeal protein",
        type="retrieval",
        modality=Modality.PROTEIN,
        datasets=[
            Dataset(
                path="tattabio/arch_retrieval",
                revision="main",
            ),
            Dataset(
                path="tattabio/arch_retrieval_qrels",
                description="Relevance between query and corpus proteins",
                revision="main",
            ),
        ],
        primary_metric_id="map_at_5",
    )

    def run(self, model: BioSeqTransformer) -> TaskResult:
        return run_retrieval_task(model, self.metadata)


class EukRetrieval(Task):
    metadata = TaskMetadata(
        id="euk_retrieval",
        display_name="Euk Retrieval",
        description="Retrieves bacterial proteins with similar swissprot annotations to a query eukaryotic protein",
        type="retrieval",
        modality=Modality.PROTEIN,
        datasets=[
            Dataset(
                path="tattabio/euk_retrieval",
                revision="main",
            ),
            Dataset(
                path="tattabio/euk_retrieval_qrels",
                description="Relevance between query and corpus proteins",
                revision="main",
            ),
        ],
        primary_metric_id="map_at_5",
    )

    def run(self, model: BioSeqTransformer) -> TaskResult:
        return run_retrieval_task(model, self.metadata)
