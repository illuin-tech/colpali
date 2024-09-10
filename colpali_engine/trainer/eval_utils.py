from typing import Dict

from mteb.evaluation.evaluators import RetrievalEvaluator


class CustomRetrievalEvaluator(RetrievalEvaluator):
    """
    Wrapper class for the MTEB retrieval evaluator.
    """

    def compute_mteb_metrics(
        self,
        relevant_docs: Dict[str, dict[str, int]],
        results: Dict[str, dict[str, float]],
        **kwargs,
    ) -> Dict[str, float]:
        """
        Compute the MTEB retrieval metrics.
        """
        ndcg, _map, recall, precision, naucs = self.evaluate(
            relevant_docs,
            results,
            self.k_values,
            ignore_identical_ids=kwargs.get("ignore_identical_ids", True),
        )

        mrr = self.evaluate_custom(relevant_docs, results, self.k_values, "mrr")

        scores = {
            **{f"ndcg_at_{k.split('@')[1]}": v for (k, v) in ndcg.items()},
            **{f"map_at_{k.split('@')[1]}": v for (k, v) in _map.items()},
            **{f"recall_at_{k.split('@')[1]}": v for (k, v) in recall.items()},
            **{f"precision_at_{k.split('@')[1]}": v for (k, v) in precision.items()},
            **{f"mrr_at_{k.split('@')[1]}": v for (k, v) in mrr[0].items()},
            **{f"naucs_at_{k.split('@')[1]}": v for (k, v) in naucs.items()},
        }
        return scores
