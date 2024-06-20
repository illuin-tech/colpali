import torch
from mteb.evaluation.evaluators import RetrievalEvaluator


class CustomEvaluator:
    def __init__(self, is_multi_vector=False):
        self.is_multi_vector = is_multi_vector
        self.mteb_evaluator = RetrievalEvaluator()

    def evaluate(self, qs, ps):
        if self.is_multi_vector:
            scores = self.evaluate_colbert(qs, ps)
        else:
            scores = self.evaluate_biencoder(qs, ps)

        assert scores.shape[0] == len(qs)

        arg_score = scores.argmax(dim=1)
        # compare to arange
        accuracy = (arg_score == torch.arange(scores.shape[0], device=scores.device)).sum().item() / scores.shape[0]
        print(arg_score)
        print(f"Top 1 Accuracy (verif): {accuracy}")

        # cast to numpy
        # scores = scores.cpu().numpy()
        scores = scores.to(torch.float32).cpu().numpy()
        return scores

    def compute_metrics(self, relevant_docs, results, **kwargs):
        # wrap mteb package

        ndcg, _map, recall, precision, naucs = self.mteb_evaluator.evaluate(
            relevant_docs,
            results,
            self.mteb_evaluator.k_values,
            ignore_identical_ids=kwargs.get("ignore_identical_ids", True),
        )
        mrr = self.mteb_evaluator.evaluate_custom(relevant_docs, results, self.mteb_evaluator.k_values, "mrr")
        scores = {
            **{f"ndcg_at_{k.split('@')[1]}": v for (k, v) in ndcg.items()},
            **{f"map_at_{k.split('@')[1]}": v for (k, v) in _map.items()},
            **{f"recall_at_{k.split('@')[1]}": v for (k, v) in recall.items()},
            **{f"precision_at_{k.split('@')[1]}": v for (k, v) in precision.items()},
            **{f"mrr_at_{k.split('@')[1]}": v for (k, v) in mrr[0].items()},
            **{f"naucs_at_{k.split('@')[1]}": v for (k, v) in naucs.items()},
        }
        return scores

    def evaluate_colbert(self, qs, ps, batch_size=128) -> torch.Tensor:
        scores = []
        for i in range(0, len(qs), batch_size):
            scores_batch = []
            qs_batch = torch.nn.utils.rnn.pad_sequence(qs[i : i + batch_size], batch_first=True, padding_value=0).to(
                "cuda"
            )
            for j in range(0, len(ps), batch_size):
                ps_batch = torch.nn.utils.rnn.pad_sequence(
                    ps[j : j + batch_size], batch_first=True, padding_value=0
                ).to("cuda")
                scores_batch.append(torch.einsum("bnd,csd->bcns", qs_batch, ps_batch).max(dim=3)[0].sum(dim=2))
            scores_batch = torch.cat(scores_batch, dim=1).cpu()
            scores.append(scores_batch)
        scores = torch.cat(scores, dim=0)
        return scores

    def evaluate_biencoder(self, qs, ps) -> torch.Tensor:

        qs = torch.stack(qs)
        ps = torch.stack(ps)

        scores = torch.einsum("bd,cd->bc", qs, ps)
        return scores
