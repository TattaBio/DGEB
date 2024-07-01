"""
Evaluator objects for different evaluation types.
"""

import logging
import random
from abc import ABC, abstractmethod
import heapq
from collections import defaultdict
import pytrec_eval
import numpy as np
import sklearn.cluster
import torch
from scipy.stats import pearsonr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    label_ranking_average_precision_score,
)
from sklearn.metrics.cluster import v_measure_score
from sklearn.metrics.pairwise import (
    paired_cosine_distances,
    paired_euclidean_distances,
    paired_manhattan_distances,
)
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import MultiLabelBinarizer
from typing import Dict, List, Tuple

from .eval_utils import (
    cos_sim,
    dot_score,
    mrr,
    recall_cap,
    hole,
    confidence_scores,
    nAUC,
    top_k_accuracy,
)


class Evaluator(ABC):
    """Base class for all evaluators
    Extend this class and implement __call__ for custom evaluators.
    """

    def __init__(self, seed=42, **kwargs):
        self.seed = seed
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

    @abstractmethod
    def __call__(self, model):
        """This is called during training to evaluate the model.
        It returns scores.

        Parameters
        ----------
        model:
            the model to evaluate
        """
        pass


logger = logging.getLogger(__name__)


class logRegClassificationEvaluator(Evaluator):
    def __init__(
        self,
        embeds_train,
        y_train,
        embeds_test,
        y_test,
        max_iter=1000,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embeds_train = embeds_train
        self.y_train = y_train
        self.embeds_test = embeds_test
        self.y_test = y_test

        self.max_iter = max_iter

    def __call__(self):
        scores = {}
        clf = LogisticRegression(
            random_state=self.seed,
            n_jobs=-1,
            max_iter=self.max_iter,
            verbose=1 if logger.isEnabledFor(logging.DEBUG) else 0,
        )
        logger.info(f"Encoding {len(self.embeds_train)} training embeds...")
        X_train = np.asarray(self.embeds_train)

        logger.info(f"Encoding {len(self.embeds_test)} test embeds...")
        X_test = np.asarray(self.embeds_test)
        logger.info("Fitting logistic regression classifier...")
        clf.fit(X_train, self.y_train)
        logger.info("Evaluating...")
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred, average="macro")
        scores["accuracy"] = accuracy
        scores["f1"] = f1

        # if binary classification
        if len(np.unique(self.y_train)) == 2:
            ap = average_precision_score(self.y_test, y_pred)
            scores["ap"] = ap

        return scores


class ClusteringEvaluator(Evaluator):
    def __init__(
        self,
        embeds,
        labels,
        clustering_batch_size=500,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embeds = embeds
        self.labels = labels
        self.clustering_batch_size = clustering_batch_size

    def __call__(self):
        logger.info(f"Encoding {len(self.embeds)} embeds...")
        corpus_embeddings = np.asarray(self.embeds)

        logger.info("Fitting Mini-Batch K-Means model...")
        clustering_model = sklearn.cluster.MiniBatchKMeans(
            n_clusters=len(set(self.labels)),
            batch_size=self.clustering_batch_size,
            n_init="auto",
        )
        clustering_model.fit(corpus_embeddings)
        cluster_assignment = clustering_model.labels_

        logger.info("Evaluating...")
        v_measure = v_measure_score(self.labels, cluster_assignment)

        return {"v_measure": v_measure}


class PairClassificationEvaluator(Evaluator):
    """Evaluate a model based on the similarity of the embeddings by calculating the accuracy of identifying similar and
    dissimilar embeds.
    The metrics are the cosine similarity as well as euclidean and Manhattan distance
    The returned score is the accuracy with a specified metric.
    The results are written in a CSV. If a CSV already exists, then values are appended.
    The labels need to be 0 for dissimilar pairs and 1 for similar pairs.
    :param embeds1: The first column of embeds
    :param embeds2: The second column of embeds
    :param labels: labels[i] is the label for the pair (embeds1[i], embeds2[i]). Must be 0 or 1
    :param name: Name for the output
    :param write_csv: Write results to a CSV file
    """

    def __init__(self, embeds1, embeds2, labels, **kwargs):
        super().__init__(**kwargs)
        self.embeds1 = embeds1
        self.embeds2 = embeds2
        self.labels = labels

        assert len(self.embeds1) == len(self.embeds2)
        assert len(self.embeds1) == len(self.labels)
        for label in labels:
            assert label == 0 or label == 1

    def __call__(self):
        scores = self.compute_metrics()
        # Compute the max of Average Precision (AP) over all distance metrics.
        top_ap_score = max(score for k, score in scores.items() if k.endswith("_ap"))
        scores["top_ap"] = top_ap_score
        return scores

    def compute_metrics(self):
        embeddings1 = np.array(self.embeds1)
        embeddings2 = np.array(self.embeds2)

        logger.info("Computing similarity distances...")
        cosine_scores = 1 - paired_cosine_distances(embeddings1, embeddings2)
        manhattan_distances = paired_manhattan_distances(embeddings1, embeddings2)
        euclidean_distances = paired_euclidean_distances(embeddings1, embeddings2)

        embeddings1_np = np.asarray(embeddings1)
        embeddings2_np = np.asarray(embeddings2)
        dot_scores = [
            np.dot(embeddings1_np[i], embeddings2_np[i])
            for i in range(len(embeddings1_np))
        ]

        logger.info("Computing metrics...")
        labels = np.asarray(self.labels)
        output_scores = {}
        for short_name, name, scores, reverse in [
            ["cos_sim", "Cosine-Similarity", cosine_scores, True],
            ["manhattan", "Manhattan-Distance", manhattan_distances, False],
            ["euclidean", "Euclidean-Distance", euclidean_distances, False],
            ["dot", "Dot-Product", dot_scores, True],
        ]:
            metrics = self._compute_metrics(scores, labels, reverse)
            metrics = {short_name + "_" + k: v for k, v in metrics.items()}
            output_scores.update(metrics)

        return output_scores

    @staticmethod
    def _compute_metrics(scores, labels, high_score_more_similar):
        """Compute the metrics for the given scores and labels.

        Args:
            scores (`np.ndarray` of shape (n_pairs, )): The similarity/dissimilarity scores for the pairs.
            labels (`np.ndarray` of shape (n_pairs, )): The labels for the pairs.
            high_score_more_similar (`bool`): If true, then the higher the score, the more similar the pairs are.

        Returns:
            `dict`: The metrics for the given scores and labels.
        """
        acc, acc_threshold = PairClassificationEvaluator.find_best_acc_and_threshold(
            scores, labels, high_score_more_similar
        )
        f1, precision, recall, f1_threshold = (
            PairClassificationEvaluator.find_best_f1_and_threshold(
                scores, labels, high_score_more_similar
            )
        )
        ap = PairClassificationEvaluator.ap_score(
            scores, labels, high_score_more_similar
        )

        return {
            "accuracy": acc,
            "accuracy_threshold": acc_threshold,
            "f1": f1,
            "f1_threshold": f1_threshold,
            "precision": precision,
            "recall": recall,
            "ap": ap,
        }

    @staticmethod
    def find_best_acc_and_threshold(scores, labels, high_score_more_similar: bool):
        assert len(scores) == len(labels)
        rows = list(zip(scores, labels))

        rows = sorted(rows, key=lambda x: x[0], reverse=high_score_more_similar)

        max_acc = 0
        best_threshold = -1

        positive_so_far = 0
        remaining_negatives = sum(np.array(labels) == 0)

        for i in range(len(rows) - 1):
            score, label = rows[i]
            if label == 1:
                positive_so_far += 1
            else:
                remaining_negatives -= 1

            acc = (positive_so_far + remaining_negatives) / len(labels)
            if acc > max_acc:
                max_acc = acc
                best_threshold = (rows[i][0] + rows[i + 1][0]) / 2

        return max_acc, best_threshold

    @staticmethod
    def find_best_f1_and_threshold(scores, labels, high_score_more_similar: bool):
        assert len(scores) == len(labels)

        scores = np.asarray(scores)
        labels = np.asarray(labels)

        rows = list(zip(scores, labels))

        rows = sorted(rows, key=lambda x: x[0], reverse=high_score_more_similar)

        best_f1 = best_precision = best_recall = 0
        threshold = 0
        nextract = 0
        ncorrect = 0
        total_num_duplicates = sum(labels)

        for i in range(len(rows) - 1):
            score, label = rows[i]
            nextract += 1

            if label == 1:
                ncorrect += 1

            if ncorrect > 0:
                precision = ncorrect / nextract
                recall = ncorrect / total_num_duplicates
                f1 = 2 * precision * recall / (precision + recall)
                if f1 > best_f1:
                    best_f1 = f1
                    best_precision = precision
                    best_recall = recall
                    threshold = (rows[i][0] + rows[i + 1][0]) / 2

        return best_f1, best_precision, best_recall, threshold

    @staticmethod
    def ap_score(scores, labels, high_score_more_similar: bool):
        return average_precision_score(
            labels, scores * (1 if high_score_more_similar else -1)
        )


class MultiClassMultiOutputLogRegClassificationEvaluator(Evaluator):
    def __init__(
        self,
        embeds_train,
        y_train,
        embeds_test,
        y_test,
        max_iter=1000,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embeds_train = embeds_train
        self.y_train = y_train
        self.embeds_test = embeds_test
        self.y_test = y_test
        self.max_iter = max_iter

    def __call__(self):
        scores = {}
        mlb = MultiLabelBinarizer()
        # all classes in y_train and y_test

        class_labels = list(self.y_train) + list(self.y_test)
        labels = [class_label.split(", ") for class_label in class_labels]
        mlb.fit(labels)
        train_labels = [class_label.split(", ") for class_label in self.y_train]
        test_labels = [class_label.split(", ") for class_label in self.y_test]

        y_train = mlb.transform(train_labels)
        y_test = mlb.transform(test_labels)
        clf = MultiOutputRegressor(
            LogisticRegression(
                random_state=self.seed, solver="lbfgs", max_iter=self.max_iter
            )
        ).fit(self.embeds_train, y_train)
        y_pred = clf.predict(self.embeds_test)

        results_dict = classification_report(y_test, y_pred, output_dict=True)
        assert isinstance(
            results_dict, dict
        ), "Should always be true since `output_dict=True` is passed to sklearn.metric.classification_report"
        scores["precision"] = results_dict["macro avg"]["precision"]
        scores["recall"] = results_dict["macro avg"]["recall"]
        scores["f1"] = results_dict["macro avg"]["f1-score"]
        scores["accuracy"] = accuracy_score(y_test, y_pred)

        return scores


class MultiClassMultiOutputKNNClassificationEvaluator(Evaluator):
    def __init__(
        self,
        embeds_train,
        y_train,
        embeds_test,
        y_test,
        n_neighbors=5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embeds_train = embeds_train
        self.y_train = y_train
        self.embeds_test = embeds_test
        self.y_test = y_test
        self.n_neighbors = n_neighbors

    def __call__(self):
        scores = {}

        mlb = MultiLabelBinarizer()
        class_labels = list(self.y_train) + list(self.y_test)
        labels = [class_label.split(", ") for class_label in class_labels]
        mlb.fit(labels)
        train_labels = [class_label.split(", ") for class_label in self.y_train]
        test_labels = [class_label.split(", ") for class_label in self.y_test]

        y_train = mlb.transform(train_labels)
        y_test = mlb.transform(test_labels)
        clf = sklearn.neighbors.KNeighborsClassifier(
            n_neighbors=self.n_neighbors, metric="cosine"
        )
        logger.info("Fitting KNN classifier...")
        clf.fit(self.embeds_train, y_train)
        logger.info("Evaluating...")
        y_pred = clf.predict(self.embeds_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro")
        precision = precision_score(y_test, y_pred, average="macro")
        recall = recall_score(y_test, y_pred, average="macro")
        lrap = label_ranking_average_precision_score(y_test, y_pred)
        scores["f1"] = f1
        scores["accuracy"] = accuracy
        scores["precision"] = precision
        scores["recall"] = recall
        scores["lrap"] = lrap

        return scores


class BiGeneMiningEvaluator(Evaluator):
    """
    BiGene Mining Evaluator, analogous to Bitext Mining Evaluator https://github.com/embeddings-benchmark/mteb/blob/main/mteb/evaluation/evaluators/BitextMiningEvaluator.py.

    If top_k > 1, then recall@k is also computed.
    """

    def __init__(self, embeds1, embeds2, top_k=1, **kwargs):
        super().__init__(**kwargs)
        self.n = len(embeds1)
        self.embeds1 = np.array(embeds1)
        self.embeds2 = np.array(embeds2)
        self.gold = list(zip(range(self.n), range(self.n)))
        self.top_k = top_k

    def __call__(self):
        scores = self.compute_metrics()
        return scores

    def compute_metrics(self):
        logger.info(f"Finding nearest neighbors... with top_k={self.top_k}")
        nearest_neighbors = self._similarity_search(
            self.embeds1, self.embeds2, top_k=self.top_k
        )

        # Compute errors
        logger.info("Computing metrics...")
        labels = []
        predictions = []

        # Get predictions and labels for top_k=1.
        for i, x in enumerate(nearest_neighbors):
            j = x[0]["corpus_id"]
            predictions.append(j)
            labels.append(self.gold[i][1])

        scores = {
            "precision": precision_score(
                labels, predictions, zero_division=0, average="weighted"
            ),
            "recall": recall_score(
                labels, predictions, zero_division=0, average="weighted"
            ),
            "f1": f1_score(labels, predictions, zero_division=0, average="weighted"),
            "accuracy": accuracy_score(labels, predictions),
        }

        if self.top_k > 1:
            # Compute recall@k.
            top_k_preds = []
            for i, x in enumerate(nearest_neighbors):
                top_k_preds.append([pred["corpus_id"] for pred in x])
            top_k_recall = [
                self.gold[i][1] in top_k_pred
                for i, top_k_pred in enumerate(top_k_preds)
            ]
            scores[f"recall_at_{self.top_k}"] = sum(top_k_recall) / len(top_k_recall)
        return scores

    def _similarity_search(
        self,
        query_embeddings,
        corpus_embeddings,
        query_chunk_size=100,
        corpus_chunk_size=500000,
        top_k=1,
        score_function=cos_sim,
    ):
        """This function performs a cosine similarity search between a list of query embeddings  and a list of corpus embeddings.
        It can be used for Information Retrieval / Semantic Search for corpora up to about 1 Million entries.
        :param query_embeddings: A 2 dimensional tensor with the query embeddings.
        :param corpus_embeddings: A 2 dimensional tensor with the corpus embeddings.
        :param query_chunk_size: Process 100 queries simultaneously. Increasing that value increases the speed, but requires more memory.
        :param corpus_chunk_size: Scans the corpus 50k entries at a time. Increasing that value increases the speed, but requires more memory.
        :param top_k: Retrieve top k matching entries.
        :param score_function: Function for computing scores. By default, cosine similarity.
        :return: Returns a list with one entry for each query. Each entry is a list of dictionaries with the keys 'corpus_id' and 'score', sorted by decreasing cosine similarity scores.
        """
        query_embeddings = torch.from_numpy(query_embeddings)
        corpus_embeddings = torch.from_numpy(corpus_embeddings)
        if len(query_embeddings.shape) == 1:
            query_embeddings = query_embeddings.unsqueeze(0)
        if len(corpus_embeddings.shape) == 1:
            corpus_embeddings = corpus_embeddings.unsqueeze(0)

        # Check that corpus and queries are on the same device
        if corpus_embeddings.device != query_embeddings.device:
            query_embeddings = query_embeddings.to(corpus_embeddings.device)

        queries_result_list = [[] for _ in range(len(query_embeddings))]

        for query_start_idx in range(0, len(query_embeddings), query_chunk_size):
            # Iterate over chunks of the corpus
            for corpus_start_idx in range(0, len(corpus_embeddings), corpus_chunk_size):
                # Compute cosine similarities
                cos_scores = score_function(
                    query_embeddings[
                        query_start_idx : query_start_idx + query_chunk_size
                    ],
                    corpus_embeddings[
                        corpus_start_idx : corpus_start_idx + corpus_chunk_size
                    ],
                )

                # Get top-k scores
                cos_scores_top_k_values, cos_scores_top_k_idx = torch.topk(
                    cos_scores,
                    min(top_k, len(cos_scores[0])),
                    dim=1,
                    largest=True,
                    sorted=False,
                )
                cos_scores_top_k_values = cos_scores_top_k_values.cpu().tolist()
                cos_scores_top_k_idx = cos_scores_top_k_idx.cpu().tolist()

                for query_itr in range(len(cos_scores)):
                    for sub_corpus_id, score in zip(
                        cos_scores_top_k_idx[query_itr],
                        cos_scores_top_k_values[query_itr],
                    ):
                        corpus_id = corpus_start_idx + sub_corpus_id
                        query_id = query_start_idx + query_itr
                        queries_result_list[query_id].append(
                            {"corpus_id": corpus_id, "score": score}
                        )

        # Sort and strip to top_k results
        for idx in range(len(queries_result_list)):
            queries_result_list[idx] = sorted(
                queries_result_list[idx], key=lambda x: x["score"], reverse=True
            )
            queries_result_list[idx] = queries_result_list[idx][0:top_k]

        return queries_result_list


class EDSEvaluator(Evaluator):
    """
    Evolutionary Distance Similarity Evaluator, analogous to Semantic Textual Similarity Evaluator.
    Adapted from https://github.com/embeddings-benchmark/mteb/blob/main/mteb/evaluation/evaluators/STSEvaluator.py
    """

    def __init__(self, embeds1, embeds2, gold_scores, **kwargs):
        super().__init__(**kwargs)
        self.embeds1 = embeds1
        self.embeds2 = embeds2
        self.gold_scores = gold_scores

    def __call__(self):
        embeddings1 = np.array(self.embeds1)
        embeddings2 = np.array(self.embeds2)
        logger.info("Evaluating...")
        cosine_scores = paired_cosine_distances(embeddings1, embeddings2)
        manhattan_distances = paired_manhattan_distances(embeddings1, embeddings2)
        euclidean_distances = paired_euclidean_distances(embeddings1, embeddings2)

        cosine_pearson, _ = pearsonr(self.gold_scores, cosine_scores)
        manhattan_pearson, _ = pearsonr(self.gold_scores, manhattan_distances)
        euclidean_pearson, _ = pearsonr(self.gold_scores, euclidean_distances)

        top_corr = max(
            cosine_pearson,
            manhattan_pearson,
            euclidean_pearson,
        )
        return {
            "cos_sim": cosine_pearson,
            "manhattan": manhattan_pearson,
            "euclidean": euclidean_pearson,
            "top_corr": top_corr,
        }


class RetrievalEvaluator(Evaluator):
    """Adapted from
    https://github.com/embeddings-benchmark/mteb/blob/main/mteb/evaluation/evaluators/RetrievalEvaluator.py
    """

    def __init__(
        self,
        corpus_embeds,
        query_embeds,
        corpus_ids,
        query_ids,
        qrels: Dict[str, Dict[str, int]],
        k_values: List[int] = [5, 10, 50],
        score_function: str = "cos_sim",
        corpus_chunk_size: int = 50000,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.corpus_embeds = corpus_embeds
        self.query_embeds = query_embeds
        self.corpus_ids = corpus_ids
        self.query_ids = query_ids
        self.qrels = qrels
        self.k_values = k_values
        self.top_k = max(k_values) if "top_k" not in kwargs else kwargs["top_k"]
        self.score_function = score_function
        self.score_functions = {
            "cos_sim": cos_sim,
            "dot": dot_score,
        }
        self.corpus_chunk_size = corpus_chunk_size

    def __call__(self):
        results = self.search(
            self.corpus_embeds,
            self.query_embeds,
            self.corpus_ids,
            self.query_ids,
            self.top_k,
            self.score_function,
        )
        ndcg, _map, recall, precision, naucs = self.evaluate(
            self.qrels, results, self.k_values
        )
        mrr, naucs_mrr = self.evaluate_custom(self.qrels, results, self.k_values, "mrr")
        scores = {
            **{f"ndcg_at_{k.split('@')[1]}": v for (k, v) in ndcg.items()},
            **{f"map_at_{k.split('@')[1]}": v for (k, v) in _map.items()},
            **{f"recall_at_{k.split('@')[1]}": v for (k, v) in recall.items()},
            **{f"precision_at_{k.split('@')[1]}": v for (k, v) in precision.items()},
            **{f"mrr_at_{k.split('@')[1]}": v for (k, v) in mrr.items()},
            **{
                k.replace("@", "_at_").replace("_P", "_precision").lower(): v
                for k, v in naucs.items()
            },
            **{
                k.replace("@", "_at_").replace("_P", "_precision").lower(): v
                for k, v in naucs_mrr.items()
            },
        }
        return scores

    def search(
        self,
        corpus_embeds,
        query_embeds,
        corpus_ids,
        query_ids,
        top_k: int,
        score_function: str,
        return_sorted: bool = False,
        **kwargs,
    ) -> dict[str, dict[str, float]]:
        # Create embeddings for all queries using model.encode()
        # Runs semantic search against the corpus embeddings
        # Returns a ranked list with the corpus ids
        if score_function not in self.score_functions:
            raise ValueError(
                f"score function: {score_function} must be either (cos_sim) for cosine similarity or (dot) for dot product"
            )
        # make query embeds and corpus embeds torch tensors
        query_embeds = torch.from_numpy(query_embeds)
        corpus_embeds = torch.from_numpy(corpus_embeds)
        itr = range(0, len(corpus_embeds), self.corpus_chunk_size)
        results = defaultdict(dict)
        # Keep only the top-k docs for each query
        result_heaps = defaultdict(list)
        for batch_num, corpus_start_idx in enumerate(itr):
            logger.info("Searching Batch {}/{}...".format(batch_num + 1, len(itr)))
            corpus_end_idx = min(
                corpus_start_idx + self.corpus_chunk_size, len(corpus_ids)
            )
            sub_corpus_embeds = corpus_embeds[corpus_start_idx:corpus_end_idx]
            # Compute similarites using either cosine-similarity or dot product
            cos_scores = self.score_functions[score_function](
                query_embeds, sub_corpus_embeds
            )
            cos_scores[torch.isnan(cos_scores)] = -1

            # Get top-k values
            cos_scores_top_k_values, cos_scores_top_k_idx = torch.topk(
                cos_scores,
                min(
                    top_k + 1,
                    len(cos_scores[1]) if len(cos_scores) > 1 else len(cos_scores[-1]),
                ),
                dim=1,
                largest=True,
                sorted=return_sorted,
            )
            cos_scores_top_k_values = cos_scores_top_k_values.cpu().tolist()
            cos_scores_top_k_idx = cos_scores_top_k_idx.cpu().tolist()

            for query_itr in range(len(query_embeds)):
                query_id = query_ids[query_itr]
                for sub_corpus_id, score in zip(
                    cos_scores_top_k_idx[query_itr], cos_scores_top_k_values[query_itr]
                ):
                    corpus_id = corpus_ids[corpus_start_idx + sub_corpus_id]
                    if corpus_id != query_id:
                        if len(result_heaps[query_id]) < top_k:
                            # Push item on the heap
                            heapq.heappush(result_heaps[query_id], (score, corpus_id))
                        else:
                            # If item is larger than the smallest in the heap, push it on the heap then pop the smallest element
                            heapq.heappushpop(
                                result_heaps[query_id], (score, corpus_id)
                            )

        for qid in result_heaps:
            for score, corpus_id in result_heaps[qid]:
                results[qid][corpus_id] = score

        return results

    @staticmethod
    def evaluate(
        qrels: dict[str, dict[str, int]],
        results: dict[str, dict[str, float]],
        k_values: List[int],
        ignore_identical_ids: bool = True,
    ) -> Tuple[Dict[str, float], dict[str, float], dict[str, float], dict[str, float]]:
        if ignore_identical_ids:
            logger.info(
                "For evaluation, we ignore identical query and document ids (default), please explicitly set ``ignore_identical_ids=False`` to ignore this."
            )
            popped = []
            for qid, rels in results.items():
                for pid in list(rels):
                    if qid == pid:
                        results[qid].pop(pid)
                        popped.append(pid)

        all_ndcgs, all_aps, all_recalls, all_precisions = {}, {}, {}, {}

        for k in k_values:
            all_ndcgs[f"NDCG@{k}"] = []
            all_aps[f"MAP@{k}"] = []
            all_recalls[f"Recall@{k}"] = []
            all_precisions[f"P@{k}"] = []

        map_string = "map_cut." + ",".join([str(k) for k in k_values])
        ndcg_string = "ndcg_cut." + ",".join([str(k) for k in k_values])
        recall_string = "recall." + ",".join([str(k) for k in k_values])
        precision_string = "P." + ",".join([str(k) for k in k_values])
        evaluator = pytrec_eval.RelevanceEvaluator(
            qrels, {map_string, ndcg_string, recall_string, precision_string}
        )
        scores = evaluator.evaluate(results)

        for query_id in scores.keys():
            for k in k_values:
                all_ndcgs[f"NDCG@{k}"].append(scores[query_id]["ndcg_cut_" + str(k)])
                all_aps[f"MAP@{k}"].append(scores[query_id]["map_cut_" + str(k)])
                all_recalls[f"Recall@{k}"].append(scores[query_id]["recall_" + str(k)])
                all_precisions[f"P@{k}"].append(scores[query_id]["P_" + str(k)])
        ndcg, _map, recall, precision = (
            all_ndcgs.copy(),
            all_aps.copy(),
            all_recalls.copy(),
            all_precisions.copy(),
        )

        for k in k_values:
            ndcg[f"NDCG@{k}"] = round(sum(ndcg[f"NDCG@{k}"]) / len(scores), 5)
            _map[f"MAP@{k}"] = round(sum(_map[f"MAP@{k}"]) / len(scores), 5)
            recall[f"Recall@{k}"] = round(sum(recall[f"Recall@{k}"]) / len(scores), 5)
            precision[f"P@{k}"] = round(sum(precision[f"P@{k}"]) / len(scores), 5)
        naucs = RetrievalEvaluator.evaluate_abstention(
            results, {**all_ndcgs, **all_aps, **all_recalls, **all_precisions}
        )
        return ndcg, _map, recall, precision, naucs

    @staticmethod
    def evaluate_abstention(
        results: dict[str, dict[str, float]],
        metric_scores: dict[str, list[float]],
    ) -> Dict[str, float]:
        """Computes normalized Area Under the Curve on a set of evaluated instances as presented in the paper https://arxiv.org/abs/2402.12997"""
        all_sim_scores = [list(results[qid].values()) for qid in list(results.keys())]
        all_conf_scores = [
            confidence_scores(sim_scores) for sim_scores in all_sim_scores
        ]
        conf_fcts = list(all_conf_scores[0].keys())
        all_conf_scores = {
            fct: np.array([x[fct] for x in all_conf_scores]) for fct in conf_fcts
        }
        metric_scores = {k: np.array(v) for k, v in metric_scores.items()}
        naucs = {}

        for metric_name, scores in metric_scores.items():
            for fct, conf_scores in all_conf_scores.items():
                naucs[f"nAUC_{metric_name}_{fct}"] = nAUC(conf_scores, scores)

        return naucs

    @staticmethod
    def evaluate_custom(
        qrels: dict[str, dict[str, int]],
        results: dict[str, dict[str, float]],
        k_values: List[int],
        metric: str,
        output_type: str = "all",
    ) -> Tuple[Dict[str, float]]:
        if metric.lower() in ["mrr", "mrr@k", "mrr_cut"]:
            metric_scores = mrr(qrels, results, k_values, output_type)

        elif metric.lower() in ["recall_cap", "r_cap", "r_cap@k"]:
            metric_scores = recall_cap(qrels, results, k_values, output_type)

        elif metric.lower() in ["hole", "hole@k"]:
            metric_scores = hole(qrels, results, k_values, output_type)

        elif metric.lower() in [
            "acc",
            "top_k_acc",
            "accuracy",
            "accuracy@k",
            "top_k_accuracy",
        ]:
            metric_scores = top_k_accuracy(qrels, results, k_values, output_type)

        naucs = RetrievalEvaluator.evaluate_abstention(results, metric_scores)
        metric_scores_avg = {k: sum(v) / len(v) for k, v in metric_scores.items()}

        return metric_scores_avg, naucs
