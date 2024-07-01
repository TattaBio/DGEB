import logging
import re
from abc import ABC, abstractmethod
from functools import partial
from types import SimpleNamespace
from typing import Dict, List, Literal, Optional

import numpy as np
import torch
import tqdm as tqdm
from datasets import Dataset
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoTokenizer,
    BatchEncoding,
    DefaultDataCollator,
    T5EncoderModel,
    T5Tokenizer,
)
from transformers.modeling_outputs import BaseModelOutput

from .modality import Modality
from .eval_utils import ForwardHook, pool

logger = logging.getLogger(__name__)


class BioSeqTransformer(ABC):
    """
    Abstract class to wrap models which map biological sequences (DNA/Prot) to embeddings.
    Modelled after SentenceTransformer (https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/SentenceTransformer.py)

    Args:
        model_name: Name or path to the pretrained model.
        layers: List of model layers to probe. Can be integers or "mid" or "last".
        devices: List of device ids for inference. If cuda is not available, will use cpu.
        num_processes: Number of processes to use for data loading.
        max_seq_length: Maximum sequence length of the input sequences.
        l2_norm: If true, embeddings are L2-normalized before they are returned.
        batch_size: Batch size for encoding.
        pool_type: Pooling strategy to use. One of "mean", "max", "cls", "last".
    """

    def __init__(
        self,
        model_name: str,
        layers: Optional[List[int] | Literal["mid"] | Literal["last"]] = None,
        devices: List[int] = [0],
        num_processes: int = 16,
        max_seq_length: int = 1024,
        l2_norm: bool = False,
        batch_size: int = 128,
        pool_type: str = "mean",
    ):
        super().__init__()

        self.id = self.__class__.__name__
        self.hf_name = model_name
        self.encoder = self._load_model(model_name)
        if not hasattr(self.encoder, "config"):
            raise ValueError(
                'The model from `self._load_model()` must have a "config" attribute.'
            )
        self.config = self.encoder.config
        self.tokenizer = self._get_tokenizer(model_name)
        self.num_param = sum(p.numel() for p in self.encoder.parameters())
        self.data_collator = DefaultDataCollator()
        self.gpu_count = len(devices)
        self.l2_norm = l2_norm

        self.device = torch.device(
            f"cuda:{devices[0]}" if torch.cuda.is_available() else "cpu"
        )
        self.num_processes = num_processes
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.pool_type = pool_type

        if self.gpu_count > 1:
            self.encoder = torch.nn.DataParallel(self.encoder, device_ids=devices)
        self.encoder.to(self.device)
        self.encoder.eval()

        mid_layer = self.num_layers // 2
        last_layer = self.num_layers - 1
        mid_layer_label = f"mid ({mid_layer})"
        last_layer_label = f"last ({self.num_layers - 1})"

        if layers is None:
            logger.debug(f"Using default layers: {mid_layer_label}, {last_layer_label}")
            self.layers = [mid_layer, last_layer]
            self.layer_labels = [mid_layer_label, last_layer_label]
        elif layers == "mid":
            self.layers = [mid_layer]
            self.layer_labels = [mid_layer_label]
        elif layers == "last":
            self.layers = [last_layer]
            self.layer_labels = [last_layer_label]
        else:
            self.layers = layers
            self.layer_labels = [str(layer) for layer in layers]

    def _encode_single_batch(self, batch_dict: Dict[str, Tensor]):
        """Returns the output embedding for the given batch with shape [batch, num_layers, D]."""
        outputs = self.encoder(**batch_dict, output_hidden_states=True)
        embeds = [outputs.hidden_states[layer] for layer in self.layers]
        embeds = [
            pool(layer_embeds, batch_dict["attention_mask"], self.pool_type)
            for layer_embeds in embeds
        ]
        # Stack with shape [B, num_layers, D].
        embeds = torch.stack(embeds, dim=1)
        return embeds

    def _load_model(self, model_name):
        return AutoModel.from_pretrained(model_name, trust_remote_code=True)

    def _get_tokenizer(self, model_name):
        return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    def _tokenize_func(
        self, tokenizer, examples: Dict[str, List], max_seq_length: int
    ) -> BatchEncoding:
        batch_dict = tokenizer(
            examples["input_seqs"],
            max_length=max_seq_length,
            padding=True,
            truncation=True,
        )
        return batch_dict

    @property
    def metadata(self) -> Dict:
        return {
            "hf_name": self.hf_name,
            "revision": "...",  # TODO: Fix
            "num_layers": self.num_layers,
            "num_params": self.num_param,
            "embed_dim": self.embed_dim,
        }

    @property
    @abstractmethod
    def num_layers(self) -> int:
        pass

    @property
    @abstractmethod
    def embed_dim(self) -> int:
        pass

    @property
    @abstractmethod
    def modality(self) -> Modality:
        pass

    @torch.no_grad()
    def encode(self, sequences, **kwargs) -> np.ndarray:
        """Returns a list of embeddings for the given sequences.
        Args:
            sequences (`List[str]`): List of sequences to encode
        Returns:
            `np.ndarray`: Embeddings for the given sequences of shape [num_sequences, num_layers, embedding_dim].
        """
        dataset = Dataset.from_dict({"input_seqs": sequences})
        dataset.set_transform(
            partial(
                self._tokenize_func, self.tokenizer, max_seq_length=self.max_seq_length
            )
        )
        data_loader = DataLoader(
            dataset,
            batch_size=self.batch_size * self.gpu_count,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_processes,
            collate_fn=self.data_collator,
            pin_memory=True,
        )

        if max(self.layers) >= self.num_layers:
            raise ValueError(
                f"Layer {max(self.layers)} is not available in the model. Choose a layer between 0 and {self.num_layers - 1}"
            )

        encoded_embeds = []
        for batch_dict in tqdm.tqdm(
            data_loader, desc="encoding", mininterval=10, disable=len(sequences) < 128
        ):
            batch_dict = {k: v.to(self.device) for k, v in batch_dict.items()}

            embeds = self._encode_single_batch(batch_dict)

            if self.l2_norm:
                embeds = F.normalize(embeds, p=2, dim=-1)
            encoded_embeds.append(embeds.cpu().numpy())

        return np.concatenate(encoded_embeds, axis=0)


class ESM(BioSeqTransformer):
    """ESM model from https://huggingface.co/docs/transformers/en/model_doc/esm"""

    MODEL_NAMES = [
        "facebook/esm2_t6_8M_UR50D",
        "facebook/esm2_t12_35M_UR50D",
        "facebook/esm2_t30_150M_UR50D",
        "facebook/esm2_t33_650M_UR50D",
        "facebook/esm2_t36_3B_UR50D",
        "facebook/esm2_t48_15B_UR50D",
    ]

    @property
    def modality(self) -> Modality:
        return Modality.PROTEIN

    @property
    def num_layers(self) -> int:
        return self.config.num_hidden_layers

    @property
    def embed_dim(self) -> int:
        return self.config.hidden_size


class ESM3(BioSeqTransformer):
    """ESM3 model from https://github.com/evolutionaryscale/esm"""

    MODEL_NAMES = ["esm3_sm_open_v1"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Register forward hooks to store embeddings per layer.
        self.hooks = [
            ForwardHook(self.encoder.transformer.blocks[layer]) for layer in self.layers
        ]

    @property
    def modality(self) -> Modality:
        return Modality.PROTEIN

    @property
    def num_layers(self) -> int:
        return self.config.num_hidden_layers

    @property
    def embed_dim(self) -> int:
        return self.config.hidden_size

    def _load_model(self, model_name):
        try:
            from esm.models.esm3 import ESM3 as ModelESM3
        except ImportError:
            raise ImportError(
                "ESM3 is not installed. Please install it with `pip install esm`."
            )
        model = ModelESM3.from_pretrained("esm3_sm_open_v1")
        model.config = SimpleNamespace(
            num_hidden_layers=len(model.transformer.blocks),
            hidden_size=model.transformer.blocks[0].ffn[-1].out_features,
        )
        return model

    def _get_tokenizer(self, model_name):
        try:
            from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer
        except ImportError:
            raise ImportError(
                "ESM3 is not installed. Please install it with `pip install esm`."
            )
        return EsmSequenceTokenizer()

    def _encode_single_batch(self, batch_dict: Dict[str, Tensor]):
        _ = self.encoder.forward(sequence_tokens=batch_dict["input_ids"])
        embeds = [hook.output for hook in self.hooks]
        embeds = [
            pool(layer_embeds, batch_dict["attention_mask"], self.pool_type)
            for layer_embeds in embeds
        ]
        # Stack with shape [B, num_layers, D].
        embeds = torch.stack(embeds, dim=1)
        embeds = embeds.to(torch.float32)
        return embeds


class ProtT5(BioSeqTransformer):
    """ProtT5 model from https://github.com/agemagician/ProtTrans"""

    MODEL_NAMES = [
        "Rostlab/prot_t5_xl_uniref50",
        "Rostlab/prot_t5_xl_bfd",
        "Rostlab/prot_t5_xxl_uniref50",
        "Rostlab/prot_t5_xxl_bfd",
    ]

    @property
    def modality(self) -> Modality:
        return Modality.PROTEIN

    @property
    def num_layers(self) -> int:
        return self.config.num_layers

    @property
    def embed_dim(self) -> int:
        return self.config.d_model

    def _load_model(self, model_name):
        return T5EncoderModel.from_pretrained(model_name)

    def _get_tokenizer(self, model_name):
        return T5Tokenizer.from_pretrained(model_name, do_lower_case=False)

    def _tokenize_func(
        self, tokenizer, examples: Dict[str, List], max_seq_length: int
    ) -> BatchEncoding:
        example_sequences = examples["input_seqs"]
        # Add space between amino acids to make sure they are tokenized correctly.
        example_sequences = [" ".join(sequence) for sequence in example_sequences]
        example_sequences = [
            re.sub(r"[UZOB]", "X", sequence) for sequence in example_sequences
        ]
        batch_dict = tokenizer(
            example_sequences,
            max_length=max_seq_length,
            padding=True,
            truncation=True,
            add_special_tokens=True,
        )

        return batch_dict


class ProGen(BioSeqTransformer):
    """ProGen models from https://github.com/salesforce/progen."""

    MODEL_NAMES = [
        "hugohrban/progen2-small",
        "hugohrban/progen2-medium",
        "hugohrban/progen2-base",
        "hugohrban/progen2-large",
        "hugohrban/progen2-xlarge",
    ]

    @property
    def modality(self) -> Modality:
        return Modality.PROTEIN

    @property
    def num_layers(self) -> int:
        return self.config.n_layer

    @property
    def embed_dim(self) -> int:
        return self.config.embed_dim

    def _load_model(self, model_name):
        return AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

    def _get_tokenizer(self, model_name_or_path):
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, trust_remote_code=True
        )
        tokenizer.pad_token = "<|pad|>"
        return tokenizer

    def _encode_single_batch(self, batch_dict: Dict[str, Tensor]):
        """Returns the output embedding for the given batch with shape [batch, num_layers, D]."""
        outputs: BaseModelOutput = self.encoder(
            input_ids=batch_dict["input_ids"],
            output_hidden_states=True,
            use_cache=False,
        )
        embeds = [outputs.hidden_states[layer] for layer in self.layers]
        embeds = [
            pool(layer_embeds, batch_dict["attention_mask"], self.pool_type)
            for layer_embeds in embeds
        ]
        # Stack with shape [B, num_layers, D].
        embeds = torch.stack(embeds, dim=1)
        return embeds


class EvoModel(BioSeqTransformer):
    """https://github.com/evo-design/evo."""

    MODEL_NAMES = [
        "togethercomputer/evo-1-8k-base",
        "togethercomputer/evo-1-131k-base",
    ]

    @property
    def modality(self) -> Modality:
        return Modality.DNA

    @property
    def num_layers(self) -> int:
        return self.config.num_layers

    @property
    def embed_dim(self) -> int:
        return self.config.hidden_size

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Register forward hooks to store embeddings per layer.
        self.hooks = []
        for layer in self.layers:
            # For the last layer, get the output of `backbone.norm`, which directly precedes `backbone.unembed`.
            # This is equivalent to the approach in https://github.com/evo-design/evo/issues/32.
            if layer == self.num_layers - 1 or layer == -1:
                self.hooks.append(ForwardHook(self.encoder.backbone.norm))
            else:
                self.hooks.append(ForwardHook(self.encoder.backbone.blocks[layer]))

    def _load_model(self, model_name):
        config = AutoConfig.from_pretrained(
            model_name, trust_remote_code=True, revision="1.1_fix"
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name, config=config, trust_remote_code=True, revision="1.1_fix"
        )
        return model

    def _get_tokenizer(self, model_name):
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, revision="1.1_fix", trust_remote_code=True
        )
        # Evo tokenizer is missing pad_token by default.
        tokenizer.add_special_tokens({"pad_token": "N"})
        return tokenizer

    def _encode_single_batch(self, batch_dict: Dict[str, Tensor]):
        _ = self.encoder(batch_dict["input_ids"], use_cache=False)
        embeds = [hook.output for hook in self.hooks]
        # The hook output for Evo middle layers is a tuple (embedding, inference_params=None).
        embeds = [x[0] if isinstance(x, tuple) else x for x in embeds]
        embeds = [
            pool(layer_embeds, batch_dict["attention_mask"], self.pool_type)
            for layer_embeds in embeds
        ]
        # Stack with shape [B, num_layers, D].
        embeds = torch.stack(embeds, dim=1)
        embeds = embeds.to(torch.float32)
        return embeds


class NTModel(BioSeqTransformer):
    """Nucleotide Transformer https://github.com/instadeepai/nucleotide-transformer"""

    MODEL_NAMES = [
        "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species",
        "InstaDeepAI/nucleotide-transformer-v2-100m-multi-species",
        "InstaDeepAI/nucleotide-transformer-v2-250m-multi-species",
        "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species",
        "InstaDeepAI/nucleotide-transformer-2.5b-multi-species",
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_seq_length = self.tokenizer.model_max_length

    @property
    def modality(self) -> Modality:
        return Modality.DNA

    @property
    def num_layers(self) -> int:
        return self.config.num_hidden_layers

    @property
    def embed_dim(self) -> int:
        return self.config.hidden_size

    def _load_model(self, model_name):
        return AutoModelForMaskedLM.from_pretrained(model_name, trust_remote_code=True)
