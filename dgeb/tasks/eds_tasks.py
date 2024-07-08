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
        raise ValueError(
            "Phylogeny tasks require 2 datasets: sequences and distances.")

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
                revision="b833ef8d8d873ea5387540562873f41d073d3e03",
            ),
            Dataset(
                path="tattabio/rpob_bac_phylogeny_distances",
                revision="0594e1501ac9fd0e3de49257b8ec318c2a0ea6f7",
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
                revision="10de75b9f5ad12340d629fd1ad015ef4319d6ee4",
            ),
            Dataset(
                path="tattabio/rpob_arch_phylogeny_distances",
                revision="2a585f0e135fe74b8ae6d31e7801c6031b0dcc18",
            ),
        ],
        primary_metric_id="top_corr",
    )

    def run(self, model: BioSeqTransformer) -> TaskResult:
        return run_eds_task(model, self.metadata)


class RpobBacDNAPhylogeny(Task):
    metadata = TaskMetadata(
        id="rpob_bac_dna_phylogeny",
        display_name="RpoB Bacterial Phylogeny",
        description="Evaluate on RpoB phylogeny distance correlation task for Bacterial DNA sequences.",
        type="eds",
        modality=Modality.DNA,
        datasets=[
            Dataset(
                path="tattabio/rpob_bac_dna_phylogeny_sequences",
                revision="8e137d3fb8886d8739ce08d1918745444c7d30d6",
            ),
            Dataset(
                path="tattabio/rpob_bac_dna_phylogeny_distances",
                revision="67339e271b2a1602208153d53d70d35ba6fa8876",
            ),
        ],
        primary_metric_id="top_corr",
    )

    def run(self, model: BioSeqTransformer) -> TaskResult:
        return run_eds_task(model, self.metadata)


class RpobArchDNAPhylogeny(Task):
    metadata = TaskMetadata(
        id="rpob_arch_dna_phylogeny",
        display_name="RpoB Archaeal Phylogeny",
        description="Evaluate on RpoB phylogeny distance correlation task for Archaeal DNA sequences.",
        type="eds",
        modality=Modality.DNA,
        datasets=[
            Dataset(
                path="tattabio/rpob_arch_dna_phylogeny_sequences",
                revision="4453552a0e1021fee8697c71a559f4d3f6da2408",
            ),
            Dataset(
                path="tattabio/rpob_arch_dna_phylogeny_distances",
                revision="51df97684a927ec2203568e80175ef26a62db039",
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
                revision="bce06d79d9ce58413e7bcbed6943905d1afb8b26",
            ),
            Dataset(
                path="tattabio/fefe_phylogeny_distances",
                revision="d6357cee9b4071a8dcdeef54083006f0d5e94fd2",
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
                revision="efde1456b86748909cbcfecb07d783756d570aa3",
            ),
            Dataset(
                path="tattabio/bac_16S_distances",
                revision="5c8ba5dfa600bb930d34af2fbc2b17f0acab62d3",
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
                revision="e0f0b5d5bd4b08a329b08c2bf4cc800781dff7f0",
            ),
            Dataset(
                path="tattabio/arch_16S_distances",
                revision="b0356b632a954be70cefd57e3a02e7e1ccd34408",
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
                revision="5174cb3b2c5c46b61307fd1c2c08f5c432655196",
            ),
            Dataset(
                path="tattabio/euk_18S_distances",
                revision="c4cea4fbb1185d08e0e01fd28ffb8b06a25025da",
            ),
        ],
        primary_metric_id="top_corr",
    )

    def run(self, model: BioSeqTransformer) -> TaskResult:
        return run_eds_task(model, self.metadata)
