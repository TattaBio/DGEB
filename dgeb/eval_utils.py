"""Utility functions for evaluation."""

from typing import Any, Dict, List, Tuple
import json
import torch
import random
import numpy as np
from sklearn.metrics import auc


class ForwardHook:
    """Pytorch forward hook class to store outputs of intermediate layers."""

    def __init__(self, module: torch.nn.Module):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.output = None

    def hook_fn(self, module, input, output):
        self.output = output

    def close(self):
        self.hook.remove()


def pool(
    last_hidden_states: torch.Tensor, attention_mask: torch.Tensor, pool_type: str
) -> torch.Tensor:
    """Pool embeddings across the sequence length dimension."""
    assert (
        last_hidden_states.ndim == 3
    ), f"Expected hidden_states to have shape [batch, seq_len, D], got shape: {last_hidden_states.shape}"
    assert (
        attention_mask.ndim == 2
    ), f"Expected attention_mask to have shape [batch, seq_len], got shape: {attention_mask.shape}"
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    if pool_type == "mean":
        emb = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    elif pool_type == "max":
        emb = last_hidden.max(dim=1)[0]
    elif pool_type == "cls":
        emb = last_hidden[:, 0]
    elif pool_type == "last":
        emb = last_hidden[torch.arange(last_hidden.size(0)), attention_mask.sum(1) - 1]
    else:
        raise ValueError(f"pool_type {pool_type} not supported")
    return emb


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def write_results_to_json(results: Dict[str, Any], results_path: str):
    """Write results dict to a json file."""
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)


def merge_split_elem_embeds(ids, embeds, preserve_order: bool = False):
    """Merge embeddings with the same id by mean-pooling and optionally preserve order in which they appear.

    Args:
        ids: Array of string ids, [batch].
        embeds: Array of embeddings, [batch, ...].

    Returns:
        ids: Unique ids, [unique_batch].
        embeds: Array of embeddings, [unique_batch, ...].
    """
    unique_ids, indices = np.unique(ids, return_inverse=True)
    shape_no_batch = embeds.shape[1:]
    sums = np.zeros([unique_ids.size, *shape_no_batch], dtype=embeds.dtype)
    counts = np.bincount(indices, minlength=unique_ids.size)
    np.add.at(sums, indices, embeds)
    # Add trailing dimensions to counts.
    counts = counts[(...,) + (None,) * len(shape_no_batch)]
    mean_pooled = sums / counts
    # Preserve the order of the input ids.
    if preserve_order:
        order = []
        for id in unique_ids:
            idx = np.where(ids == id)[0][0]
            order.append(idx)
        re_order = np.argsort(order)
        unique_ids = unique_ids[re_order]
        mean_pooled = mean_pooled[re_order]
    return unique_ids, mean_pooled


def paired_dataset(labels, embeds):
    """Creates a paired dataset for consecutive operonic gene pairs."""
    embeds1 = embeds[:-1]
    embeds2 = embeds[1:]
    labels = labels[:-1]
    return embeds1, embeds2, labels


def cos_sim(a, b):
    """Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.

    Return:
        Matrix with res[i][j]  = cos_sim(a[i], b[j])
    """  # noqa: D402
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    return torch.mm(a_norm, b_norm.transpose(0, 1))


def dot_score(a: torch.Tensor, b: torch.Tensor):
    """Computes the dot-product dot_prod(a[i], b[j]) for all i and j.
    :return: Matrix with res[i][j]  = dot_prod(a[i], b[j])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    return torch.mm(a, b.transpose(0, 1))


# From https://github.com/beir-cellar/beir/blob/f062f038c4bfd19a8ca942a9910b1e0d218759d4/beir/retrieval/custom_metrics.py#L4
def mrr(
    qrels: dict[str, dict[str, int]],
    results: dict[str, dict[str, float]],
    k_values: List[int],
    output_type: str = "mean",
) -> Tuple[Dict[str, float]]:
    MRR = {}

    for k in k_values:
        MRR[f"MRR@{k}"] = []

    k_max, top_hits = max(k_values), {}

    for query_id, doc_scores in results.items():
        top_hits[query_id] = sorted(
            doc_scores.items(), key=lambda item: item[1], reverse=True
        )[0:k_max]

    for query_id in top_hits:
        query_relevant_docs = set(
            [doc_id for doc_id in qrels[query_id] if qrels[query_id][doc_id] > 0]
        )
        for k in k_values:
            rr = 0
            for rank, hit in enumerate(top_hits[query_id][0:k]):
                if hit[0] in query_relevant_docs:
                    rr = 1.0 / (rank + 1)
                    break
            MRR[f"MRR@{k}"].append(rr)

    if output_type == "mean":
        for k in k_values:
            MRR[f"MRR@{k}"] = round(sum(MRR[f"MRR@{k}"]) / len(qrels), 5)

    elif output_type == "all":
        pass

    return MRR


# From https://github.com/embeddings-benchmark/mteb/blob/8178981fd8fcd546d7031afe61a083d13c41520f/mteb/evaluation/evaluators/utils.py
def recall_cap(
    qrels: dict[str, dict[str, int]],
    results: dict[str, dict[str, float]],
    k_values: List[int],
    output_type: str = "mean",
) -> Tuple[Dict[str, float]]:
    capped_recall = {}

    for k in k_values:
        capped_recall[f"R_cap@{k}"] = []

    k_max = max(k_values)

    for query_id, doc_scores in results.items():
        top_hits = sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)[
            0:k_max
        ]
        query_relevant_docs = [
            doc_id for doc_id in qrels[query_id] if qrels[query_id][doc_id] > 0
        ]
        for k in k_values:
            retrieved_docs = [
                row[0] for row in top_hits[0:k] if qrels[query_id].get(row[0], 0) > 0
            ]
            denominator = min(len(query_relevant_docs), k)
            capped_recall[f"R_cap@{k}"].append(len(retrieved_docs) / denominator)

    if output_type == "mean":
        for k in k_values:
            capped_recall[f"R_cap@{k}"] = round(
                sum(capped_recall[f"R_cap@{k}"]) / len(qrels), 5
            )

    elif output_type == "all":
        pass

    return capped_recall


# From https://github.com/embeddings-benchmark/mteb/blob/8178981fd8fcd546d7031afe61a083d13c41520f/mteb/evaluation/evaluators/utils.py
def hole(
    qrels: dict[str, dict[str, int]],
    results: dict[str, dict[str, float]],
    k_values: List[int],
    output_type: str = "mean",
) -> Tuple[Dict[str, float]]:
    Hole = {}

    for k in k_values:
        Hole[f"Hole@{k}"] = []

    annotated_corpus = set()
    for _, docs in qrels.items():
        for doc_id, score in docs.items():
            annotated_corpus.add(doc_id)

    k_max = max(k_values)

    for _, scores in results.items():
        top_hits = sorted(scores.items(), key=lambda item: item[1], reverse=True)[
            0:k_max
        ]
        for k in k_values:
            hole_docs = [
                row[0] for row in top_hits[0:k] if row[0] not in annotated_corpus
            ]
            Hole[f"Hole@{k}"].append(len(hole_docs) / k)

    if output_type == "mean":
        for k in k_values:
            Hole[f"Hole@{k}"] = round(Hole[f"Hole@{k}"] / len(qrels), 5)

    elif output_type == "all":
        pass

    return Hole


# From https://github.com/embeddings-benchmark/mteb/blob/8178981fd8fcd546d7031afe61a083d13c41520f/mteb/evaluation/evaluators/utils.py
def top_k_accuracy(
    qrels: dict[str, dict[str, int]],
    results: dict[str, dict[str, float]],
    k_values: List[int],
    output_type: str = "mean",
) -> Tuple[Dict[str, float]]:
    top_k_acc = {}

    for k in k_values:
        top_k_acc[f"Accuracy@{k}"] = []

    k_max, top_hits = max(k_values), {}

    for query_id, doc_scores in results.items():
        top_hits[query_id] = [
            item[0]
            for item in sorted(
                doc_scores.items(), key=lambda item: item[1], reverse=True
            )[0:k_max]
        ]

    for query_id in top_hits:
        query_relevant_docs = set(
            [doc_id for doc_id in qrels[query_id] if qrels[query_id][doc_id] > 0]
        )
        for k in k_values:
            for relevant_doc_id in query_relevant_docs:
                if relevant_doc_id in top_hits[query_id][0:k]:
                    top_k_acc[f"Accuracy@{k}"].append(1.0)
                    break

    if output_type == "mean":
        for k in k_values:
            top_k_acc[f"Accuracy@{k}"] = round(
                top_k_acc[f"Accuracy@{k}"] / len(qrels), 5
            )

    elif output_type == "all":
        pass

    return top_k_acc


# From https://github.com/embeddings-benchmark/mteb/blob/8178981fd8fcd546d7031afe61a083d13c41520f/mteb/evaluation/evaluators/utils.py
def confidence_scores(sim_scores: List[float]) -> Dict[str, float]:
    """Computes confidence scores for a single instance = (query, positives, negatives)

    Args:
        sim_scores: Query-documents similarity scores with length `num_pos+num_neg`

    Returns:
        conf_scores:
            - `max`: Maximum similarity score
            - `std`: Standard deviation of similarity scores
            - `diff1`: Difference between highest and second highest similarity scores
    """
    sim_scores_sorted = sorted(sim_scores)[::-1]

    cs_max = sim_scores_sorted[0]
    cs_std = np.std(sim_scores)
    if len(sim_scores) > 1:
        cs_diff1 = sim_scores_sorted[0] - sim_scores_sorted[1]
    elif len(sim_scores) == 1:
        cs_diff1 = 0.0

    conf_scores = {"max": cs_max, "std": cs_std, "diff1": cs_diff1}

    return conf_scores


# From https://github.com/embeddings-benchmark/mteb/blob/8178981fd8fcd546d7031afe61a083d13c41520f/mteb/evaluation/evaluators/utils.py
def nAUC(
    conf_scores: np.ndarray,
    metrics: np.ndarray,
    abstention_rates: np.ndarray = np.linspace(0, 1, 11)[:-1],
) -> float:
    """Computes normalized Area Under the Curve on a set of evaluated instances as presented in the paper https://arxiv.org/abs/2402.12997
    1/ Computes the raw abstention curve, i.e., the average evaluation metric at different abstention rates determined by the confidence scores
    2/ Computes the oracle abstention curve, i.e., the best theoretical abstention curve (e.g.: at a 10% abstention rate, the oracle abstains on the bottom-10% instances with regard to the evaluation metric)
    3/ Computes the flat abstention curve, i.e., the one remains flat for all abstention rates (ineffective abstention)
    4/ Computes the area under the three curves
    5/ Finally scales the raw AUC between the oracle and the flat AUCs to get normalized AUC

    Args:
        conf_scores: Instance confidence scores used for abstention thresholding, with shape `(num_test_instances,)`
        metrics: Metric evaluations at instance-level (e.g.: average precision, NDCG...), with shape `(num_test_instances,)`
        abstention_rates: Target rates for the computation of the abstention curve

    Returns:
        abst_nauc: Normalized area under the abstention curve (upper-bounded by 1)
    """

    def abstention_curve(
        conf_scores: np.ndarray,
        metrics: np.ndarray,
        abstention_rates: np.ndarray = np.linspace(0, 1, 11)[:-1],
    ) -> np.ndarray:
        """Computes the raw abstention curve for a given set of evaluated instances and corresponding confidence scores

        Args:
            conf_scores: Instance confidence scores used for abstention thresholding, with shape `(num_test_instances,)`
            metrics: Metric evaluations at instance-level (e.g.: average precision, NDCG...), with shape `(num_test_instances,)`
            abstention_rates: Target rates for the computation of the abstention curve

        Returns:
            abst_curve: Abstention curve of length `len(abstention_rates)`
        """
        conf_scores_argsort = np.argsort(conf_scores)
        abst_curve = np.zeros(len(abstention_rates))

        for i, rate in enumerate(abstention_rates):
            num_instances_abst = min(
                round(rate * len(conf_scores_argsort)), len(conf_scores) - 1
            )
            abst_curve[i] = metrics[conf_scores_argsort[num_instances_abst:]].mean()

        return abst_curve

    abst_curve = abstention_curve(conf_scores, metrics, abstention_rates)
    or_curve = abstention_curve(metrics, metrics, abstention_rates)
    abst_auc = auc(abstention_rates, abst_curve)
    or_auc = auc(abstention_rates, or_curve)
    flat_auc = or_curve[0] * (abstention_rates[-1] - abstention_rates[0])

    if or_auc == flat_auc:
        abst_nauc = np.nan
    else:
        abst_nauc = (abst_auc - flat_auc) / (or_auc - flat_auc)

    return abst_nauc
