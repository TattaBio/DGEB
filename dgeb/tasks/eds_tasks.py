"""
Evolutionary Distance Similarity (EDS) tasks compare embedding distances to continuous evolutionary distances.
The label distances are typically derived from phylogenetic trees.
"""

import logging
from collections import defaultdict

import numpy as np
import pandas as pd

from dgeb.evaluators import EDSEvaluator
from dgeb.modality import Modality
from dgeb.models import BioSeqTransformer
from dgeb.tasks import Dataset, Task, TaskMetadata, TaskResult

logger = logging.getLogger(__name__)


def run_eds_task(model: BioSeqTransformer, metadata: TaskMetadata) -> TaskResult:
    """Evaluate phylogeny distance correlation task. Utilizes the Evolutionary Distance Similarity (EDS) evaluator."""
    if len(metadata.datasets) != 2:
        raise ValueError("Phylogeny tasks require 2 datasets: sequences and distances.")

    ds = metadata.datasets[0].load()["train"]
    distance_df = metadata.datasets[1].load()["train"].to_pandas()
    assert isinstance(
        distance_df, pd.DataFrame
    ), f"Expected DataFrame, got {type(distance_df)}"

    id_index_dict = {k: i for i, k in enumerate(ds["Entry"])}
    distance_df["embeds1"] = None
    distance_df["embeds2"] = None
    test_embeds = model.encode(ds["Sequence"])
    layer_results = defaultdict(dict)
    for i, layer in enumerate(model.layers):
        for row_idx, row in distance_df.iterrows():
            id1 = row["ID1"]
            id2 = row["ID2"]
            embedding1 = test_embeds[id_index_dict[id1], i]
            embedding2 = test_embeds[id_index_dict[id2], i]
            distance_df.at[row_idx, "embeds1"] = embedding1
            distance_df.at[row_idx, "embeds2"] = embedding2
        embeds1 = np.array(distance_df["embeds1"].tolist())
        embeds2 = np.array(distance_df["embeds2"].tolist())
        dists = np.array(distance_df["distance"].tolist())
        evaluator = EDSEvaluator(embeds1, embeds2, dists)
        layer_results["layers"][layer] = evaluator()
        # log results
        logger.info(
            f"Layer: {layer}, {metadata.display_name} distance correlation results: {layer_results['layers'][layer]}"
        )

    return TaskResult.from_dict(metadata, layer_results, model.metadata)


class RpobBacPhylogeny(Task):
    metadata = TaskMetadata(
        id="rpob_bac_phylogeny",
        display_name="RpoB Bacterial Phylogeny",
        description="Evaluate on RpoB phylogeny distance correlation task for Bacterial sequences.",
        type="eds",
        modality=Modality.PROTEIN,
        datasets=[
            Dataset(
                path="tattabio/rpob_bac_phylogeny_sequences",
                revision="main",
            ),
            Dataset(
                path="tattabio/rpob_bac_phylogeny_distances",
                revision="main",
            ),
        ],
        primary_metric_id="top_corr",
    )

    def run(self, model: BioSeqTransformer) -> TaskResult:
        return run_eds_task(model, self.metadata)


class RpobArchPhylogeny(Task):
    metadata = TaskMetadata(
        id="rpob_arch_phylogeny",
        display_name="RpoB Archaeal Phylogeny",
        description="Evaluate on RpoB phylogeny distance correlation task for Archaeal sequences.",
        type="eds",
        modality=Modality.PROTEIN,
        datasets=[
            Dataset(
                path="tattabio/rpob_arch_phylogeny_sequences",
                revision="main",
            ),
            Dataset(
                path="tattabio/rpob_arch_phylogeny_distances",
                revision="main",
            ),
        ],
        primary_metric_id="top_corr",
    )

    def run(self, model: BioSeqTransformer) -> TaskResult:
        return run_eds_task(model, self.metadata)


class FeFePhylogeny(Task):
    metadata = TaskMetadata(
        id="fefe_phylogeny",
        display_name="FeFeHydrogenase Phylogeny",
        description="Evaluate on FeFeHydrogenase phylogeny distance correlation task.",
        type="eds",
        modality=Modality.PROTEIN,
        datasets=[
            Dataset(
                path="tattabio/fefe_phylogeny_sequences",
                revision="main",
            ),
            Dataset(
                path="tattabio/fefe_phylogeny_distances",
                revision="main",
            ),
        ],
        primary_metric_id="top_corr",
    )

    def run(self, model: BioSeqTransformer) -> TaskResult:
        return run_eds_task(model, self.metadata)


class Bac16SPhylogeny(Task):
    metadata = TaskMetadata(
        id="bac_16S_phylogeny",
        display_name="16S Bacterial Phylogeny",
        description="Evaluate on 16S Bacterial phylogeny distance correlation task.",
        type="eds",
        modality=Modality.DNA,
        datasets=[
            Dataset(
                path="tattabio/bac_16S_sequences",
                revision="main",
            ),
            Dataset(
                path="tattabio/bac_16S_distances",
                revision="main",
            ),
        ],
        primary_metric_id="top_corr",
    )

    def run(self, model: BioSeqTransformer) -> TaskResult:
        return run_eds_task(model, self.metadata)


class Arch16SPhylogeny(Task):
    metadata = TaskMetadata(
        id="arch_16S_phylogeny",
        display_name="16S Archaeal Phylogeny",
        description="Evaluate on 16S Archaeal phylogeny distance correlation task.",
        type="eds",
        modality=Modality.DNA,
        datasets=[
            Dataset(
                path="tattabio/arch_16S_sequences",
                revision="main",
            ),
            Dataset(
                path="tattabio/arch_16S_distances",
                revision="main",
            ),
        ],
        primary_metric_id="top_corr",
    )

    def run(self, model: BioSeqTransformer) -> TaskResult:
        return run_eds_task(model, self.metadata)


class Euk18SPhylogeny(Task):
    metadata = TaskMetadata(
        id="euk_18S_phylogeny",
        display_name="18S Eukaryotic Phylogeny",
        description="Evaluate on 18S Eukaryotic phylogeny distance correlation task.",
        type="eds",
        modality=Modality.DNA,
        datasets=[
            Dataset(
                path="tattabio/euk_18S_sequences",
                revision="main",
            ),
            Dataset(
                path="tattabio/euk_18S_distances",
                revision="main",
            ),
        ],
        primary_metric_id="top_corr",
    )

    def run(self, model: BioSeqTransformer) -> TaskResult:
        return run_eds_task(model, self.metadata)
