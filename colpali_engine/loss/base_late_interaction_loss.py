import torch


class BaseColbertLoss(torch.nn.Module):
    """
    Base class for ColBERT loss functions (late-interaction).
    """

    @staticmethod
    def compute_colbert_scores(qs: torch.Tensor, ps: torch.Tensor) -> torch.Tensor:
        """
        Compute the ColBERT scores between the query (`qs`) and document (`ps`, p for passage) embeddings.
        Both the query and document embeddings should have the same batch size.

        Args:
        - qs [queries] (n_queries, num_query_tokens, dim)
        - ps [documents] (n_documents, num_doc_tokens, dim)

        Returns:
        - torch.Tensor: The ColBERT scores with shape (n_queries, n_documents).
        """

        scores = torch.einsum("bnd,csd->bcns", qs, ps).max(dim=3)[0].sum(dim=2)  # (n_queries, n_documents)

        return scores
