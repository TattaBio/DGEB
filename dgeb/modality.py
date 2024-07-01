"""Defines the data modality enum."""

from enum import Enum


class Modality(Enum):
    """Data modality, either DNA or protein sequence."""

    PROTEIN = "protein"
    DNA = "dna"
