"""
Classification tasks take in biological sequence and functional labels.
Multi-class and/or multi-label classification tasks are supported.
"""

import logging
from collections import defaultdict

import datasets
import numpy as np

from dgeb.eval_utils import merge_split_elem_embeds
from dgeb.evaluators import (
    MultiClassMultiOutputKNNClassificationEvaluator,
    logRegClassificationEvaluator,
)
from dgeb.modality import Modality
from dgeb.models import BioSeqTransformer
from dgeb.tasks import Dataset, Task, TaskMetadata, TaskResult

logger = logging.getLogger(__name__)


def split_sequences(
    ds: datasets.DatasetDict, max_seq_length: int
) -> datasets.DatasetDict:
    """Split sequences into chunks of max_seq_length using datasets.Dataset.map()."""

    def _split_sequence(examples, max_seq_length):
        assert (
            len(examples["Sequence"]) == 1
        ), "split map function should use batch size of 1."
        example = {k: v[0] for k, v in examples.items()}
        seq = example["Sequence"]
        # Split by chunks of max_seq_length.
        seq_split = [
            seq[i : i + max_seq_length] for i in range(0, len(seq), max_seq_length)
        ]
        # Repeat other fields by the number of splits.
        example = {
            k: [v] * len(seq_split) for k, v in example.items() if k != "Sequence"
        }
        example["Sequence"] = seq_split
        return example

    ds = ds.map(
        _split_sequence,
        batched=True,
        batch_size=1,
        fn_kwargs={"max_seq_length": max_seq_length},
        keep_in_memory=True,
        load_from_cache_file=False,
    )
    return ds


def run_classification_task(
    model: BioSeqTransformer, metadata: TaskMetadata
) -> TaskResult:
    """Evaluate on classification tasks using logistic regression classifier."""
    ds = metadata.datasets[0].load()
    layer_results = defaultdict(dict)
    train_embeds = model.encode(ds["train"]["Sequence"])
    test_embeds = model.encode(ds["test"]["Sequence"])
    for i, layer in enumerate(model.layers):
        layer_results["layers"][layer] = logRegClassificationEvaluator(
            train_embeds[:, i],
            ds["train"]["Label"],
            test_embeds[:, i],
            ds["test"]["Label"],
        )()
        logger.info(
            f"Layer: {layer}, {metadata.display_name} results: {layer_results['layers'][layer]}"
        )
    return TaskResult.from_dict(metadata, layer_results, model.metadata)


class EnzymeCommissionClassification(Task):
    metadata = TaskMetadata(
        id="ec_classification",
        display_name="EC Classification",
        description="Evaluate on Enzyme Commission number classification task.",
        type="classification",
        modality=Modality.PROTEIN,
        datasets=[
            Dataset(
                path="tattabio/ec_classification",
                revision="ead5570168e6969a5149f6861e8a33d6b5d22498",
            )
        ],
        primary_metric_id="f1",
    )

    def run(self, model: BioSeqTransformer) -> TaskResult:
        return run_classification_task(model, self.metadata)


class EnzymeCommissionDNAClassification(Task):
    metadata = TaskMetadata(
        id="ec_dna_classification",
        display_name="EC Classification",
        description="Evaluate on Enzyme Commission number classification task using DNA sequences.",
        type="classification",
        modality=Modality.DNA,
        datasets=[
            Dataset(
                path="tattabio/ec_classification_dna",
                revision="cd61c74b4930cf9f1963e6d73ff7f14e2c8e74dd",
            )
        ],
        primary_metric_id="f1",
    )

    def run(self, model: BioSeqTransformer) -> TaskResult:
        return run_classification_task(model, self.metadata)


class ConvergentEnzymesClassification(Task):
    metadata = TaskMetadata(
        id="convergent_enzymes_classification",
        display_name="Convergent Enzymes Classification",
        description="Evaluate on convergent enzymes classification task, where convergent enzymes are proteins with the same EC number but without blastp hits against each other",
        type="classification",
        modality=Modality.PROTEIN,
        datasets=[
            Dataset(
                path="tattabio/convergent_enzymes",
                revision="37f75609f54de2bc0911ccb72faf1c2f5a4285aa",
            )
        ],
        primary_metric_id="f1",
    )

    def run(self, model: BioSeqTransformer) -> TaskResult:
        return run_classification_task(model, self.metadata)


def run_mibig_task(model: BioSeqTransformer, metadata: TaskMetadata) -> TaskResult:
    """
    Evaluate on MIBIG classification tasks. Multiclass, multi-label KNN classification is used for evaluation.
    """
    ds = metadata.datasets[0].load()
    if metadata.modality == Modality.DNA:
        # MIBiG DNA sequences can be very long. Instead of truncating to max_seq_length,
        # split into multiple sequences and mean pool the resulting embeddings.
        ds = split_sequences(ds, model.max_seq_length)

    layer_results = defaultdict(dict)
    train_embeds = model.encode(ds["train"]["Sequence"])
    test_embeds = model.encode(ds["test"]["Sequence"])

    train_ids = ds["train"]["Entry"]
    test_ids = ds["test"]["Entry"]
    train_labels = ds["train"]["class"]
    test_labels = ds["test"]["class"]
    train_id_to_label = {id: label for id, label in zip(train_ids, train_labels)}
    test_id_to_label = {id: label for id, label in zip(test_ids, test_labels)}
    # Mean pool embeds with the same ID.
    train_ids, train_embeds = merge_split_elem_embeds(train_ids, train_embeds)
    test_ids, test_embeds = merge_split_elem_embeds(test_ids, test_embeds)
    # Gather the labels after merging by unique ID.
    train_labels = np.array([train_id_to_label[id] for id in train_ids])
    test_labels = np.array([test_id_to_label[id] for id in test_ids])

    for i, layer in enumerate(model.layers):
        evaluator = MultiClassMultiOutputKNNClassificationEvaluator(
            train_embeds[:, i], train_labels, test_embeds[:, i], test_labels
        )
        layer_results["layers"][layer] = evaluator()
        logger.info(
            f"Layer: {layer}, MIBiG classification results: {layer_results['layers'][layer]}"
        )
    return TaskResult.from_dict(metadata, layer_results, model.metadata)


class MIBiGProteinClassification(Task):
    metadata = TaskMetadata(
        id="MIBIG_protein_classification",
        display_name="MIBiG Classification",
        description="Biosynthetic Gene cluster classification using protein sequences on MIBIG dataset.",
        type="classification",
        modality=Modality.PROTEIN,
        datasets=[
            Dataset(
                path="tattabio/mibig_classification_prot",
                revision="915a7ff28dc9820e35c4d7fd03d4c8c44a88ff1f",
            )
        ],
        primary_metric_id="f1",
    )

    def run(self, model: BioSeqTransformer) -> TaskResult:
        return run_mibig_task(model, self.metadata)


class MIBiGDNAClassification(Task):
    metadata = TaskMetadata(
        id="MIBIG_dna_classification",
        display_name="MIBiG Classification",
        description="Biosynthetic Gene cluster classification using DNA sequences on MIBIG dataset.",
        type="classification",
        modality=Modality.DNA,
        datasets=[
            Dataset(
                path="tattabio/mibig_classification_dna",
                revision="b5ca7a76d469e4e66c46f1b655903972571e6b61",
            )
        ],
        primary_metric_id="f1",
    )

    def run(self, model: BioSeqTransformer) -> TaskResult:
        return run_mibig_task(model, self.metadata)
